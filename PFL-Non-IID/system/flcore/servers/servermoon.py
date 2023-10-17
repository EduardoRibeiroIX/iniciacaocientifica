from flcore.clients.clientmoon import clientMOON
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import time


class MOON(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMOON)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def treinamento(self, i, clientes):
        s_t = time.time()
        self.selected_clients = clientes
        self.send_models()

        if i%self.eval_gap == 0:
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            self.evaluate()

        for client in self.selected_clients:
            client.train()

        self.ids.extend(self.receive_models())  #retorna os ids dos clientes ativos
        
        if self.dlg_eval and i%self.dlg_gap == 0:
            self.call_dlg(i)
        
        self.users += self.aggregate_parameters()

        self.Budget.append(time.time() - s_t)
        print('-'*50, self.Budget[-1])


    def train(self):
        k = 4
        for i in range(self.global_rounds+1):
            if i == 0:
                self.treinamento(i, self.select_clients())
                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break
            
                df_clientes = self.csv_clients(self.users)
                df_cluster_clientes = self.data_clusters(df_clientes, k)
                # print(df_cluster_clientes)
            
            else:
                clientes_cluster = self.clientes_cluster(df_cluster_clientes, self.obj_clients)
                self.selected_clients = list(clientes_cluster[1].values())
                df_clientes = self.csv_clients(self.users)
                df = self.data_clusters(df_clientes, k)
                self.users = []
                self.treinamento(i, self.selected_clients)
                df = self.updated_data(df, clientes_cluster[0], self.users)
                df_cluster_clientes = self.data_clusters(df, k)
                self.users = [df_cluster_clientes['Media_clients'].tolist()]
                # print(df_cluster_clientes)
                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest local accuracy.")
        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientMOON)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
