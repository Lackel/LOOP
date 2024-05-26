from model import CLBert
from init_parameter import init_model
from dataloader import Data
from mtp import PretrainModelManager
from utils.tools import *
from utils.memory import MemoryBank, fill_memory_bank
from utils.neighbor_dataset import NeighborsDataset
from model import BertForModel
from transformers import logging, WEIGHTS_NAME
import warnings
from scipy.spatial import distance as dist
import openai
warnings.filterwarnings('ignore')
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LoopModelManager:
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)
        self.args = args
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.model = CLBert(args.bert_model, device=self.device, num_labels=data.n_known_cls)

        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        
        if pretrained_model is None:
            pretrained_model = BertForModel(args.pretrain_dir, num_labels=data.n_known_cls)
            # if os.path.exists(args.pretrain_dir):
            #     pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model
        
        self.load_pretrained_model()
        
        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data) 
        else:
            self.num_labels = data.num_labels
        
        self.num_train_optimization_steps = int(len(data.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs
        
        self.optimizer, self.scheduler = self.get_optimizer(args)
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)

    def get_neighbor_dataset(self, args, data, indices, query_index, pred):
        dataset = NeighborsDataset(args, data.train_semi_dataset, indices, query_index, pred)
        self.train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
        self.dataset = dataset

    def get_neighbor_inds(self, args, data, km):
        memory_bank = MemoryBank(len(data.train_semi_dataset), args.feat_dim, len(data.all_label_list), 0.1)
        fill_memory_bank(data.train_semi_dataloader, self.model, memory_bank)
        indices, query_index = memory_bank.mine_nearest_neighbors(args.topk, km.labels_, km.cluster_centers_)
        return indices, query_index
    
    def get_adjacency(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1 # if in neighbors
                # if (targets[b1] == targets[b2]) and (targets[b1]>=0) and (targets[b2]>=0):
                if (targets[b1] == targets[b2]) and (inds[b1] <= args.num_labeled_examples) and (inds[b2] <= args.num_labeled_examples):
                    adj[b1][b2] = 1 # if same labels
                    # this is useful only when both have labels
        return adj

    def evaluation(self, args, data, save_results=True, plot_cm=True):
        """final clustering evaluation on test set"""
        # get features
        feats_test, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats_test = feats_test.cpu().numpy()

        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats_test)
        y_pred = km.labels_
        y_true = labels.cpu().numpy()
        results = clustering_score(y_true, y_pred, data.known_lab)
        print('results',results)
        self.test_results = results
        
        # save results
        if save_results:
            self.save_results(args)

    def train(self, args, data):
        if isinstance(self.model, nn.DataParallel):
            criterion = self.model.module.loss_cl
            ce = self.model.module.loss_ce
        else:
            criterion = self.model.loss_cl
            ce = self.model.loss_ce
        feats, labels = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
        
        # load neighbors for the first epoch
        indices, query_index = self.get_neighbor_inds(args, data, km)
        self.get_neighbor_dataset(args, data, indices, query_index, km.labels_)
        labelediter = iter(data.train_labeled_dataloader)

        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for _, batch in enumerate(self.train_dataloader):
                # 1. load data
                anchor = tuple(t.to(self.device) for t in batch["anchor"]) # anchor data
                neighbor = tuple(t.to(self.device) for t in batch["neighbor"]) # neighbor data
                pos_neighbors = batch["possible_neighbors"] # all possible neighbor inds for anchor
                data_inds = batch["index"] # neighbor data ind

                # 2. get adjacency matrix
                adjacency = self.get_adjacency(args, data_inds, pos_neighbors, batch["target"]) # (bz,bz)

                # 3. get augmentations
                if args.view_strategy == "rtr":
                    X_an = {"input_ids":self.generator.random_token_replace(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":self.generator.random_token_replace(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                elif args.view_strategy == "shuffle":
                    X_an = {"input_ids":self.generator.shuffle_tokens(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":self.generator.shuffle_tokens(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                elif args.view_strategy == "none":
                    X_an = {"input_ids":anchor[0], "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":neighbor[0], "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                else:
                    raise NotImplementedError(f"View strategy {args.view_strategy} not implemented!")
                
                # 4. compute loss and update parameters
                with torch.set_grad_enabled(True):
                    f_pos = torch.stack([self.model(X_an)["features"], self.model(X_ng)["features"]], dim=1)
                    loss_cl = criterion(f_pos, mask=adjacency, temperature=args.temp)
                    
                    try:
                        batch = labelediter.next()
                    except StopIteration:
                        labelediter = iter(data.train_labeled_dataloader)
                        batch = labelediter.next()
                    batch = tuple(t.to(self.device) for t in batch)
                    X_an = {"input_ids":batch[0], "attention_mask":batch[1], "token_type_ids":batch[2]}

                    logits = self.model(X_an)["logits"]
                    loss_ce = ce(logits, batch[3]) 

                    loss = 0.5 * loss_ce + loss_cl
                    tr_loss += loss.item()
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    nb_tr_examples += anchor[0].size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            self.dataset.count = 0
                        
            # update neighbors every several epochs
            if ((epoch + 1) % args.update_per_epoch) == 0 and ((epoch + 1) != int(args.num_train_epochs)):
                self.evaluation(args, data, save_results=False, plot_cm=False)

                feats, labels = self.get_features_labels(data.train_semi_dataloader, self.model, args)
                feats = feats.cpu().numpy()
                # k-means clustering
                km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
                indices, query_index = self.get_neighbor_inds(args, data, km)
                self.get_neighbor_dataset(args, data, indices, query_index, km.labels_)

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion*self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler
    
    def load_pretrained_model(self):
        """load the backbone of pretrained model"""
        if isinstance(self.pretrained_model, nn.DataParallel):
            pretrained_dict = self.pretrained_model.module.backbone.state_dict()
        else:
            pretrained_dict = self.pretrained_model.backbone.state_dict()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.backbone.load_state_dict(pretrained_dict, strict=False)
        else:
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for _, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature = model(X, output_hidden_states=True)["hidden_states"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels
            
    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.topk, args.view_strategy, args.seed]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'topk', 'view_strategy', 'seed']
        vars_dict = {k:v for k,v in zip(names, var)}
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            df1 = df1.append(new,ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)
    
    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def cluster_name(self, args, data):
        feats_label, labels = self.get_features_labels(data.train_labeled_dataloader, self.model, args)
        feats_label = feats_label.cpu().numpy()
        labels = labels.cpu().numpy()
        [rows, cols] = feats_label.shape
        num = np.zeros(data.n_known_cls)
        # labeled prototypes
        proto_l = np.zeros((data.n_known_cls, args.feat_dim))
        for i in range(rows):
            proto_l[labels[i]] += feats_label[i]
            num[labels[i]] += 1
        for i in range(data.n_known_cls):
            proto_l[i] = proto_l[i] / num[i]

        feats_gpu, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats_gpu.cpu().numpy()
        
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
        # unlabeled prototypes
        proto_u = km.cluster_centers_
        distance = dist.cdist(proto_l, proto_u, 'euclidean')
        _, col_ind = linear_sum_assignment(distance)
        novel_id = [i for i in range(self.num_labels) if i not in col_ind]
        cluster_centers = torch.tensor(km.cluster_centers_[novel_id])
        dis = self.EuclideanDistances(feats_gpu.cpu(), cluster_centers).T
        _, index = torch.sort(dis, dim=1)
        index = index[:, :3]
        cluster_name = []
        for i in range(len(index)):
            query = []
            for j in index[i]:
                query.append(data.train_semi_dataset.__getitem__(j)[0])
            cluster_name.append(self.query_llm(query))
        print(cluster_name)
            
    def query_llm(self, a):
        s1 = self.tokenizer.decode(a[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        s2 = self.tokenizer.decode(a[1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        s3 = self.tokenizer.decode(a[2], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        prompt = "Given the following customer utterances, return a word or a phrase to summarize the common intent of these utterances without explanation. \n Utterance 1: " + s1 + "\n Utterance 2: " + s2 + "\n Utterance 3: " + s3

        openai.api_key = self.args.api_key
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            # time.sleep(1)
            return completion.choices[0].message['content']
        except Exception:
            return Exception
                
    def EuclideanDistances(self, a, b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

    def predict_k(self, args, data):
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model.cuda(), args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels * 0.9
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num',num_labels)

        return num_labels
    
if __name__ == '__main__':

    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)
    if os.path.exists(args.pretrain_dir):
        args.disable_pretrain = True # disable internal pretrain
    else:
        args.disable_pretrain = False

    if not args.disable_pretrain:
        print('Pre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args, data)
        print('Pre-training finished!')
        manager = LoopModelManager(args, data, manager_p.model)
    else:
        manager = LoopModelManager(args, data)
    
    if args.report_pretrain:
        method = args.method
        args.method = 'pretrain'
        manager.evaluation(args, data) # evaluate when report performance on pretrain
        args.method = method

    print('Training begin...')
    manager.train(args,data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished!')
    manager.cluster_name(args,data)
    print('Saving Model ...')
    if args.save_model:
        manager.model.save_backbone(args.save_model_path)
    print("Finished!")