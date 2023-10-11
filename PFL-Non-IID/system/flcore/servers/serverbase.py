import sys

import pandas as pd
import torch
import os
import numpy
import numpy as np
import h5py
import copy
import time
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.users_0 = []
        self.users = [] # contém os dados dos usuário
        self.ids = [] # contém os ids dos clientes a cada round de traino
        self.obj_clients = {} # contém os objetos da clase Client
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        
        for i, objeto in enumerate(selected_clients):
            nome = f'{i+1}_cliente'
            self.obj_clients[nome] = objeto
        
        return list(self.obj_clients.values())
    
    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


######################################################################################


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = self.selected_clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
                
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        
        return active_clients


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        add_pr = []
        self.global_model = copy.deepcopy(self.uploaded_models[0])

        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            add_pr += self.add_parameters(w, client_model)
        add_pr = [self.valueOfList(add_pr)]

        return add_pr 


    def add_parameters(self, w, client_model):
        vp = np.array([])
        valores = []
        media = []

        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            valores.extend(self.valueOfList(client_param.data))
            server_param.data += client_param.data.clone() * w

        vp = np.append(vp, valores)
        media.append(float(vp.sum() / len(vp)))
        # print(media)
        # sys.exit()
        
        return media
    


#-------------------------------- My functions---------------------------------------#


    def valueOfList(self, value):
        """
    Retorna uma lista que contém todos os valores inteiros e flutuantes contidos em uma estrutura de dados aninhada.

    Args:
        value (list, np.ndarray, torch.Tensor): A estrutura de dados da qual você deseja extrair valores inteiros e flutuantes.

    Returns:
        list: Uma lista contendo todos os valores inteiros e flutuantes encontrados na estrutura de dados.

    Note:
        - A função aceita estruturas de dados aninhadas (listas dentro de listas).
        - Se `value` for uma instância de `torch.Tensor`, ela será convertida em um array numpy e depois em uma lista.
        - Se `value` for uma instância de `np.ndarray`, ela será convertida em uma lista.
        - A função recursivamente percorre estruturas de dados aninhadas para extrair todos os valores inteiros e flutuantes.
        - Os valores inteiros e flutuantes extraídos são adicionados à lista `valueList`.

    Exemplos:
        >>> obj = SuaClasse()
        >>> tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> lista = [1, [2, 3.5], [[4, 5.0], 6]]
        >>> obj.valueOfList(tensor)
        [1.0, 2.0, 3.0, 4.0]
        >>> obj.valueOfList(lista)
        [1, 2, 3.5, 4, 5.0, 6]
    """

        valueList = list()
    
        if type(value) == torch.Tensor:
            value = value.cpu().numpy()
            value = value.tolist()
            self.valueOfList(value)

        if type(value) == np.ndarray:
            value = value.tolist()
            self.valueOfList(value)

        if type(value) == list and len(value) > 0:
            for i in range(len(value)):
                if type(value[i]) == list and len(value[i]) > 0:
                    valueList += self.valueOfList(value[i])

                elif type(value[i]) == int or type(value[i]) == float:
                    valueList.append(value[i])

            return valueList
        

    def csv_clients(self, lista):
        """
    Converte uma lista de dados em um DataFrame pandas para exportação em formato CSV ou DataFrame.

    Args:
        lista (list): Uma lista de dados que você deseja converter em um DataFrame.

    Returns:
        pandas.DataFrame: Um DataFrame pandas com os dados da lista.

    Note:
        - A função inverte as linhas e colunas da lista de entrada por meio da operação de transposição.
        - O DataFrame resultante é nomeado com colunas representando cada "Round" e uma coluna adicional "id".
        - A coluna "id" contém identificadores associados aos dados com base na estrutura das entradas.
        - Esta função assume que as variáveis globais self.global_rounds, self.args.num_clients, self.ids e self.obj_clients já estão definidas no objeto que chama a função.

    Exemplo:
        >>> obj = SuaClasse()
        >>> lista_de_dados = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> df_resultante = obj.csv_clients(lista_de_dados)
        >>> print(df_resultante)
           Round 0  Round 1  Round 2  id
        0        1        4        7   A
        1        2        5        8   B
        2        3        6        9   C
    """

        # Faz a operação de transposição da lista, ou seja, inverte linhas e colunas
        lista = list(map(list, zip(*lista)))
        df = pd.DataFrame(lista)
        colum_names = []

        colum_names.append(f'Media_clients')
        colum_names.append(f'id_client')

        chaves = []
        for i in range(len(self.users[0])):
            chaves.extend([chave for chave, valor in self.obj_clients.items() if valor == self.ids[i]])
        
        id = []  
        for valor in chaves:
            x = [valor] #* tam
            id.extend(x)
        
        df['id_client'] = id
        df.columns = colum_names
       
        return df


    def data_clusters(self, df, nCluster):
        """
    Realiza a análise de clustering (agrupamento) em um DataFrame de dados.

    Args:
        df (pandas.DataFrame): O DataFrame contendo os dados que você deseja clusterizar.
        nCluster (int): O número de clusters desejados para a análise de clustering.

    Returns:
        None

    Note:
        - A função renomeia o índice do DataFrame como "Index" para facilitar a manipulação.
        - Ela realiza o agrupamento dos dados com base nas colunas e calcula a média dos valores dentro de cada grupo.
        - Em seguida, normaliza os dados usando o StandardScaler.
        - Utiliza o algoritmo K-Means para realizar o clustering com o número de clusters especificado em nCluster.
        - Adiciona uma coluna 'cluster' ao DataFrame resultante, indicando a qual cluster cada ponto de dados pertence.
        - A função não salva o DataFrame resultante em um arquivo CSV; essa linha de código está comentada.
    
    Exemplo:
        >>> obj = SuaClasse()
        >>> dados = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5]})
        >>> obj.data_clusters(dados, 2)
        # Nenhum valor de retorno; a função realiza operações dentro do objeto.

    """

        df = df.rename_axis("Index")
        colunas = df.columns
        colunas = colunas.tolist()
        grouped = df.groupby(colunas).mean().reset_index()
        data_for_clustering = grouped[['Media_clients']]
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data_for_clustering)
        k = nCluster
        kmeans = KMeans(n_clusters=k, random_state=0)
        grouped['cluster'] = kmeans.fit_predict(normalized_data)
        grouped_colunas = grouped.columns
        grouped_colunas = grouped_colunas.tolist()
        x = grouped[grouped_colunas]
        # x.to_csv('./csv/clientes.csv')

        return x
   

    def clientes_cluster(self, i, k, df, objeto=dict):
        """
        Retorna um dicionário contendo os identificadores de cliente como chaves e os valores associados aos clientes no cluster especificado.

        Args:
            df (pandas.DataFrame): O DataFrame contendo os dados de clientes, incluindo uma coluna 'cluster'.
            cluster (int): O número do cluster para o qual você deseja obter os valores associados aos clientes. O valor padrão é 0.
            objeto (dict): Um dicionário contendo valores associados a clientes.

        Returns:
            dict: Um novo dicionário contendo os identificadores de cliente como chaves e os valores associados aos clientes no cluster especificado.

        Note:
            - A função filtra o DataFrame df para selecionar apenas os clientes que pertencem ao cluster especificado.
            - Em seguida, ela extrai os identificadores únicos dos clientes no cluster.
            - Utiliza o dicionário objeto para encontrar os valores associados a esses identificadores e cria um novo dicionário com os identificadores como chaves e os valores associados como valores.
            - Retorna o novo dicionário contendo os valores associados aos clientes no cluster.

        Exemplo:
            >>> obj = SuaClasse()
            >>> dados = pd.DataFrame({'id': ['A', 'B', 'C', 'D'], 'cluster': [0, 1, 0, 1]})
            >>> dicionario_global = {'A': 10, 'B': 15, 'C': 20, 'D': 25}
            >>> novo_dicionario = obj.clientes_cluster(dados, cluster=0, objeto=dicionario_global)
            >>> print(novo_dicionario)
            {'A': 10, 'C': 20}

        """
        
        contagem_clusters = df['cluster'].value_counts()
        cluster_comum = contagem_clusters.idxmax()
        freq_comum = contagem_clusters.max()
        cluster_min = contagem_clusters.idxmin()

        df1 = df[df['cluster'] == cluster_comum]
        df2 = df[df['cluster'] == cluster_min]

        df = pd.concat([df1, df2], ignore_index=True)
        x = df[df.columns[-2]].unique()
        obj = objeto
        novo_dicionario = {}

        for valor in x:
            if valor in obj:
                v = obj[valor]
                novo_dicionario[valor] = v
                
        return ([[cluster_comum, freq_comum], cluster_min], novo_dicionario)


    def clientes_cluster_max(self, i, k, df, objeto=dict):
        """
        Retorna um dicionário contendo os identificadores de cliente como chaves e os valores associados aos clientes no cluster especificado.

        Args:
            df (pandas.DataFrame): O DataFrame contendo os dados de clientes, incluindo uma coluna 'cluster'.
            cluster (int): O número do cluster para o qual você deseja obter os valores associados aos clientes. O valor padrão é 0.
            objeto (dict): Um dicionário contendo valores associados a clientes.

        Returns:
            dict: Um novo dicionário contendo os identificadores de cliente como chaves e os valores associados aos clientes no cluster especificado.

        Note:
            - A função filtra o DataFrame df para selecionar apenas os clientes que pertencem ao cluster especificado.
            - Em seguida, ela extrai os identificadores únicos dos clientes no cluster.
            - Utiliza o dicionário objeto para encontrar os valores associados a esses identificadores e cria um novo dicionário com os identificadores como chaves e os valores associados como valores.
            - Retorna o novo dicionário contendo os valores associados aos clientes no cluster.

        Exemplo:
            >>> obj = SuaClasse()
            >>> dados = pd.DataFrame({'id': ['A', 'B', 'C', 'D'], 'cluster': [0, 1, 0, 1]})
            >>> dicionario_global = {'A': 10, 'B': 15, 'C': 20, 'D': 25}
            >>> novo_dicionario = obj.clientes_cluster(dados, cluster=0, objeto=dicionario_global)
            >>> print(novo_dicionario)
            {'A': 10, 'C': 20}

        """
        
        contagem_clusters = df['cluster'].value_counts()
        cluster_comum = contagem_clusters.idxmax()
        # cluster_min = random.randint()
        df = df[df['cluster'] == cluster_comum]
        x = df[df.columns[-2]].unique()
        obj = objeto
        novo_dicionario = {}
        for valor in x:
            if valor in obj:
                v = obj[valor]
                novo_dicionario[valor] = v

        return (cluster_comum, novo_dicionario)


    def clientes_cluster_min(self, i, k, df, objeto=dict):
        contagem_clusters = df['cluster'].value_counts()
        cluster_comum = contagem_clusters.idxmin()
        df = df[df['cluster'] == cluster_comum]
        x = df[df.columns[-2]].unique()
        obj = objeto
        novo_dicionario = {}
        for valor in x:
            if valor in obj:
                v = obj[valor]
                novo_dicionario[valor] = v

        return (cluster_comum, novo_dicionario)


    def updated_data(self, df_clusterizado, nCluster, news_data):
        # print(df_clusterizado)
        # print(nCluster)
        # print(news_data)
        df_clusterizado.loc[df_clusterizado['cluster'] == nCluster[0][0], 'Media_clients'] = news_data[0][:nCluster[0][1]]
        # print(df_clusterizado)
        # print('---------------------------------------------------')
        df_clusterizado.loc[df_clusterizado['cluster'] == nCluster[1], 'Media_clients'] = news_data[0][nCluster[0][1]:]
        # print(df_clusterizado)
        # sys.exit()
        self.users = [df_clusterizado['Media_clients'].tolist()]
        return df_clusterizado[['Media_clients', 'id_client']]


#-------------------------------- xxxxxxxxxxxxxxx --------------------------------------#


######################################################################################


    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        # with open("saida.txt", "w") as arquivo:
        # # Escrever o valor da variável no arquivo
        #     arquivo.write(str(test_acc))
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        with open("saida.txt", "a") as arquivo:
        # Escrever o valor da variável no arquivo
            arquivo.write(str(test_acc) + "," + str(train_loss))
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
