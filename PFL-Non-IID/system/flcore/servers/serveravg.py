import sys
import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

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
        print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        

    def train(self):


        for i in range(self.global_rounds+1):
            if i == 0:
                self.treinamento(i, self.select_clients())
                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break
                df_clientes = self.csv_clients(self.users)
                df_cluster_clientes = self.data_clusters(df_clientes, 4)
                print(df_cluster_clientes)
                # sys.exit()
            elif i == 1:
                clientes_do_cluster_0 = self.clientes_cluster(df_cluster_clientes, 0, self.obj_clients)
                self.selected_clients = list(clientes_do_cluster_0[0].values())
                self.treinamento(i, self.selected_clients)
                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

                self.users = [clientes_do_cluster_0[1][0]]
                self.obj_clients = clientes_do_cluster_0[0]
                self.ids = list(clientes_do_cluster_0[0].values())

                # print(self.ids)
                # print(self.users)
                # print(self.obj_clients)
                # print(self.selected_clients)

                df_clientes = self.csv_clients(self.users)
                df_cluster_clientes = self.data_clusters(df_clientes, 3)
                
                
                # print('-=-' * 30)
                print(df_cluster_clientes)
                # clientes_do_cluster_1 = self.clientes_cluster(df_cluster_clientes, 0, self.obj_clients)
                # print(f'==================> cluster 0: {list(clientes_do_cluster_1[0].keys())}')
                # clientes_do_cluster_2 = self.clientes_cluster(df_cluster_clientes, 1, self.obj_clients)
                # print(f'==================> cluster 1: {list(clientes_do_cluster_2[0].keys())}')
                # print('-=-' * 30)
            elif i == 2:
                clientes_do_cluster_0 = self.clientes_cluster(df_cluster_clientes, 0, self.obj_clients)
                self.selected_clients = list(clientes_do_cluster_0[0].values())
                self.treinamento(i, self.selected_clients)
                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break
                
                self.users = [clientes_do_cluster_0[1][0]]
                self.obj_clients = clientes_do_cluster_0[0]
                self.ids = list(clientes_do_cluster_0[0].values())

                df_clientes = self.csv_clients(self.users)
                df_cluster_clientes = self.data_clusters(df_clientes, 2)

                print(df_cluster_clientes)


        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        
