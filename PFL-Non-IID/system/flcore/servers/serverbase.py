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
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import MiniBatchKMeans

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
        """
    Configura e adiciona instâncias de clientes à lista de clientes do objeto que invoca a função.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - clientObj: A classe representando um cliente no contexto específico do código.

    Atributos Esperados em self:
    - num_clients: Número total de clientes a serem configurados.
    - train_slow_clients: Lista de valores booleanos indicando se o treinamento do cliente é lento.
    - send_slow_clients: Lista de valores booleanos indicando se o envio de dados do cliente é lento.
    - dataset: Conjunto de dados para treinamento e teste.
    - args: Argumentos adicionais para inicializar uma instância de cliente.

    Retorno:
    - A função não retorna nada explicitamente, mas adiciona instâncias de clientes à lista self.clients do objeto que a chama.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função set_clients
    objeto_instancia = MinhaClasse(args, dataset, num_clients, train_slow_clients, send_slow_clients)

    # Chamando a função set_clients para configurar os clientes
    objeto_instancia.set_clients(MinhaClasseCliente)
    ```

    Notas Adicionais:
    - Certifique-se de que a classe clientObj tenha um construtor adequado que aceite os parâmetros mencionados na função.
    - A função read_client_data é usada para ler os dados do cliente a partir do conjunto de dados especificado.
    - Certifique-se de que os atributos necessários em self estejam devidamente inicializados antes de chamar a função set_clients.
        """
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
        """
    Seleciona aleatoriamente clientes lentos com base na taxa fornecida e retorna uma lista indicando quais clientes são lentos.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - slow_rate: A taxa de clientes lentos desejada, representada como um valor entre 0 e 1.

    Atributos Esperados em self:
    - num_clients: Número total de clientes disponíveis.

    Retorno:
    - slow_clients: Lista de valores booleanos indicando quais clientes são lentos, de acordo com a taxa fornecida.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função select_slow_clients
    objeto_instancia = MinhaClasse(args, dataset, num_clients)

    # Chamando a função select_slow_clients para selecionar clientes lentos com uma taxa de 0.2
    clientes_lentos = objeto_instancia.select_slow_clients(0.2)
    ```

    Notas Adicionais:
    - Certifique-se de que o atributo num_clients esteja devidamente inicializado antes de chamar a função select_slow_clients.
    - O retorno é uma lista de booleanos indicando quais clientes são lentos, onde True representa um cliente lento e False representa um cliente normal.
        """
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients


    def set_slow_clients(self):
        """
    Configura clientes lentos para treinamento e envio com base nas taxas fornecidas.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - train_slow_rate: A taxa desejada de clientes lentos durante o treinamento, representada como um valor entre 0 e 1.
    - send_slow_rate: A taxa desejada de clientes lentos durante o envio, representada como um valor entre 0 e 1.
    - train_slow_clients: Lista de valores booleanos indicando quais clientes são lentos durante o treinamento.
    - send_slow_clients: Lista de valores booleanos indicando quais clientes são lentos durante o envio.

    Retorno:
    - A função não retorna nada explicitamente, mas atualiza os atributos train_slow_clients e send_slow_clients do objeto que a chama.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função set_slow_clients
    objeto_instancia = MinhaClasse(args, dataset, num_clients, train_slow_rate=0.2, send_slow_rate=0.1)

    # Chamando a função set_slow_clients para configurar clientes lentos
    objeto_instancia.set_slow_clients()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos train_slow_rate e send_slow_rate estejam devidamente inicializados antes de chamar a função set_slow_clients.
    - A função utiliza a função select_slow_clients internamente para determinar quais clientes são lentos durante o treinamento e o envio.
    - Os resultados são refletidos nos atributos train_slow_clients e send_slow_clients do objeto que invoca a função.
        """
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)


    def select_clients(self):
        """
    Seleciona aleatoriamente um conjunto de clientes para participar de uma operação com base nas configurações atuais.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - random_join_ratio: Booleano indicando se a seleção de clientes deve ser aleatória com base em uma taxa.
    - num_join_clients: Número de clientes a serem selecionados (se random_join_ratio for False).
    - num_clients: Número total de clientes disponíveis.
    - clients: Lista de clientes disponíveis para seleção.
    - current_num_join_clients: Número atual de clientes a serem selecionados.
    - obj_clients: Dicionário para armazenar os clientes selecionados.

    Retorno:
    - Lista de clientes selecionados para participar da operação, conforme as configurações atuais.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função select_clients
    objeto_instancia = MinhaClasse(args, dataset, num_clients, random_join_ratio=True, num_join_clients=3)

    # Chamando a função select_clients para selecionar clientes
    clientes_selecionados = objeto_instancia.select_clients()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos random_join_ratio, num_join_clients e clients estejam devidamente inicializados antes de chamar a função select_clients.
    - A função utiliza np.random.choice para realizar seleções aleatórias.
    - Os clientes selecionados são armazenados no dicionário obj_clients, onde as chaves são nomes únicos gerados automaticamente.
        """
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
        """
    Envia o modelo global para os clientes participantes e registra os custos associados.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - clients: Lista de clientes participantes que receberão o modelo global.
    - global_model: Modelo global a ser enviado para os clientes.
    - send_time_cost: Dicionário para rastrear os custos associados ao envio do modelo.

    Retorno:
    - A função não retorna nada explicitamente, mas atualiza os atributos send_time_cost dos clientes participantes.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função send_models
    objeto_instancia = MinhaClasse(args, dataset, num_clients, global_model)

    # Chamando a função send_models para enviar o modelo global para os clientes
    objeto_instancia.send_models()
    ```

    Notas Adicionais:
    - Certifique-se de que o atributo clients e global_model estejam devidamente inicializados antes de chamar a função send_models.
    - A função utiliza time.time() para medir o tempo de envio e atualiza os custos no dicionário send_time_cost de cada cliente.
    - O custo total é calculado como duas vezes o tempo de envio para simular ida e volta.
        """
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


######################################################################################


    def receive_models(self):
        """
    Recebe modelos enviados por clientes selecionados com base em critérios como taxa de descarte e custo de treinamento e envio.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - selected_clients: Lista de clientes previamente selecionados para participar de uma operação.
    - client_drop_rate: Taxa de descarte de clientes durante o recebimento, representada como um valor entre 0 e 1.
    - current_num_join_clients: Número atual de clientes a serem selecionados.
    - obj_clients: Dicionário que mapeia nomes de clientes para instâncias de clientes.
    - find_key: Função que encontra a chave associada a um valor em um dicionário.
    - uploaded_ids: Lista de IDs dos clientes processados.
    - uploaded_weights: Lista de pesos normalizados associados aos clientes processados.
    - uploaded_models: Lista de modelos associados aos clientes processados.
    - time_threshold: Limiar de tempo de treinamento e envio para inclusão de um cliente.
    
    Retorno:
    - active_clients: Lista de clientes processados que atendem aos critérios especificados.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função receive_models
    objeto_instancia = MinhaClasse(args, dataset, selected_clients, client_drop_rate=0.1, time_threshold=10)

    # Chamando a função receive_models para receber modelos dos clientes selecionados
    clientes_processados = objeto_instancia.receive_models()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos selected_clients, client_drop_rate, obj_clients e time_threshold estejam devidamente inicializados antes de chamar a função receive_models.
    - A função utiliza as informações de custo de treinamento e envio de cada cliente para decidir se o cliente deve ser incluído.
    - Os resultados são refletidos nas listas uploaded_ids, uploaded_weights e uploaded_models.
        """
        assert (len(self.selected_clients) > 0)

        if self.client_drop_rate == 0.0:
            active_clients = self.selected_clients
        else:
            active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients)
            )

        print(self.find_key(self.obj_clients, active_clients))

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
        """
    Agrega os parâmetros dos modelos enviados pelos clientes selecionados, considerando seus pesos normalizados.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - uploaded_models: Lista de modelos enviados pelos clientes selecionados.
    - uploaded_weights: Lista de pesos normalizados associados aos modelos enviados.
    - global_model: Modelo global a ser atualizado com os parâmetros agregados.
    
    Retorno:
    - add_pr: Lista contendo os parâmetros agregados do modelo global.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função aggregate_parameters
    objeto_instancia = MinhaClasse(args, dataset, uploaded_models, uploaded_weights)

    # Chamando a função aggregate_parameters para agregar os parâmetros
    parametros_agregados = objeto_instancia.aggregate_parameters()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos uploaded_models, uploaded_weights e global_model estejam devidamente inicializados antes de chamar a função aggregate_parameters.
    - A função utiliza a função add_parameters internamente para agregar os parâmetros ponderados dos modelos enviados.
    - O resultado é uma lista contendo os parâmetros agregados do modelo global.
        """
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
        """
    Adiciona os parâmetros ponderados do modelo de um cliente ao modelo global.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - w: Peso normalizado associado ao modelo do cliente.
    - client_model: Modelo do cliente a ser adicionado ao modelo global.

    Atributos Esperados em self:
    - global_model: Modelo global que será atualizado com os parâmetros ponderados do modelo do cliente.

    Retorno:
    - media: Lista contendo a média dos valores dos parâmetros do modelo do cliente após a adição ponderada.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função add_parameters
    objeto_instancia = MinhaClasse(args, dataset, global_model)

    # Chamando a função add_parameters para adicionar os parâmetros ponderados do modelo do cliente
    media_valores = objeto_instancia.add_parameters(weight, client_model)
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos global_model estejam devidamente inicializados antes de chamar a função add_parameters.
    - A função realiza uma adição ponderada dos parâmetros do modelo do cliente ao modelo global.
    - O resultado é uma lista contendo a média dos valores dos parâmetros do modelo do cliente após a adição ponderada.
        """
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


    def find_key(self, objeto, lista):
        clientes = []
        for valor in lista:
            for chave, valor_objeto in objeto.items():
                if valor == valor_objeto:
                    clientes.append(chave)
        return clientes


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
        >> obj.valueOfList(tensor)
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
        df.to_csv("../clientes.csv")
        return df


    def cluster_kmeans(self, df, nCluster=int):
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
   

    def cluster_kmedoids(self, df=pd.DataFrame, nCluster=int):
        """
        Realiza a análise de clustering K-Medoids em um conjunto de dados.

        Parâmetros:
        df (DataFrame): O DataFrame contendo os dados a serem analisados. Deve conter uma coluna chamada 'Media_clients'.
        nCluster (int): O número de clusters desejado.

        Retorna:
        DataFrame: Um novo DataFrame contendo as colunas originais do DataFrame de entrada, mais uma coluna 'cluster' que indica a qual cluster cada ponto de dados pertence.

        Descrição:
        Esta função realiza a análise de clustering K-Medoids em um conjunto de dados, com o objetivo de agrupar os dados em 'nCluster' clusters. Ela segue os seguintes passos:

        1. Renomeia o índice do DataFrame para "Index".
        2. Calcula a média dos dados agrupados por todas as colunas, resultando em um novo DataFrame chamado 'grouped'.
        3. Isola a coluna 'Media_clients' para a análise de clustering.
        4. Normaliza os dados usando o StandardScaler.
        5. Executa o algoritmo K-Medoids com o número de clusters especificado.
        6. Adiciona uma coluna 'cluster' ao DataFrame 'grouped' que indica o cluster atribuído a cada ponto de dados.
        7. Retorna o DataFrame 'grouped' com as colunas originais mais a coluna 'cluster'.

        Exemplo de Uso:
        >>> from sklearn.datasets import make_blobs
        >>> import pandas as pd
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn_extra.cluster import KMedoids

        >>> # Gere um conjunto de dados de exemplo
        >>> X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
        >>> df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
        >>> df['Media_clients'] = df.mean(axis=1)

        >>> # Crie uma instância da classe que contém a função e chame a função data_kmedoids
        >>> instance = SuaClasse()
        >>> resultado = instance.data_kmedoids(df, nCluster=3)

        Neste exemplo, a função realiza uma análise de clustering K-Medoids nos dados contidos no DataFrame 'df' com 3 clusters e retorna um novo DataFrame 'resultado' com a coluna 'cluster' indicando a atribuição de cluster para cada ponto de dados.
        """

        df = df.rename_axis("Index")
        colunas = df.columns
        colunas = colunas.tolist()
        grouped = df.groupby(colunas).mean().reset_index()
        data_for_clustering = grouped[['Media_clients']]
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data_for_clustering)
        k = nCluster
        kmedoids = KMedoids(n_clusters=k, random_state=0)
        kmedoids.fit(normalized_data)
        grouped['cluster']  = kmedoids.labels_
        grouped_colunas = grouped.columns
        grouped_colunas = grouped_colunas.tolist()
        x = grouped[grouped_colunas]

        return x


    def clientes_cluster(self, df, objeto=dict):
        """
            Retorna informações sobre os clusters mais e menos frequentes em um DataFrame e um dicionário personalizado.

            Parâmetros:
            df (DataFrame): O DataFrame contendo os dados dos clientes, incluindo uma coluna 'cluster'.
            objeto (dict): Um dicionário personalizado contendo valores associados aos clusters.

            Retorna:
            list: Uma lista contendo informações sobre os clusters mais e menos frequentes e um dicionário personalizado.

            Exemplo:
            >>> df = pd.DataFrame({'cliente_id': [1, 2, 3, 4, 5],
            ...                    'cluster': ['A', 'B', 'A', 'B', 'A']})
            >>> obj = {'A': 'Alto', 'B': 'Baixo'}
            >>> clientes_cluster(df, obj)
            [['A', 3], 'B'], {'A': 'Alto', 'B': 'Baixo'}
        """
        # Calcula a contagem de clusters
        contagem_clusters = df['cluster'].value_counts()
        cluster_a = contagem_clusters.index[0]
        freq_a = contagem_clusters.get(cluster_a, 0)
        cluster_b = contagem_clusters.index[-1]

        # Separa os DataFrames para os clusters mais e menos frequentes
        df1 = df[df['cluster'] == cluster_a]
        df2 = df[df['cluster'] == cluster_b]
        df = pd.concat([df1, df2], ignore_index=True)

        # Obtém os valores únicos da coluna de cluster
        x = df[df.columns[-2]].unique()
        obj = objeto
        novo_dicionário = {}

        # Mapeia valores do objeto personalizado para os clusters
        for valor in x:
            if valor in obj:
                v = obj[valor]
                novo_dicionário[valor] = v

        return ([[cluster_a, freq_a], cluster_b], novo_dicionário)


    def clientes_cluster_random(self, df, objeto=dict):
        """
        Esta função retorna informações sobre dois clusters aleatórios e um dicionário atualizado.

        Parâmetros:
        - df (DataFrame): O DataFrame contendo os dados, incluindo uma coluna 'cluster' para agrupar os dados.
        - objeto (dict): Um dicionário contendo valores associados aos clusters.

        Retorno:
        - Uma lista contendo informações sobre dois clusters aleatórios e um novo dicionário atualizado.
        [cluster1, frequência_cluster1, cluster2], novo_dicionário

        Descrição:
        Esta função recebe um DataFrame com informações de cluster e um dicionário de objetos relacionados aos clusters.
        Ela seleciona dois clusters aleatórios, extrai informações sobre eles, cria um novo DataFrame com os dados dos clusters
        selecionados, extrai valores únicos de uma coluna no DataFrame resultante e atualiza o dicionário com valores associados a esses
        clusters. Finalmente, ela retorna as informações sobre os dois clusters e o novo dicionário.

        Exemplo de uso:
        df = pandas.DataFrame(...)
        objeto = {'cluster1': 'valor1', 'cluster2': 'valor2', ...}
        resultado = clientes_cluster_random(df, objeto)
        """
        lista_cluster = df['cluster'].unique().tolist()
        cluster_random = random.sample(lista_cluster, 2)
        contagem_clusters = df['cluster'].value_counts()
        cluster_a = cluster_random[0]
        freq_a = contagem_clusters.get(cluster_a, 0)
        cluster_b = cluster_random[1]
        df1 = df[df['cluster'] == cluster_a]
        df2 = df[df['cluster'] == cluster_b]
        df = pd.concat([df1, df2], ignore_index=True)
        x = df[df.columns[-2]].unique()
        obj = objeto
        novo_dicionario = {}

        for valor in x:
            if valor in obj:
                v = obj[valor]
                novo_dicionario[valor] = v

        return ([[cluster_a, freq_a], cluster_b], novo_dicionario)


    def updated_data(self, df_clusterizado, nCluster, news_data):
        """
    Atualiza os dados de um DataFrame 'df_clusterizado' com informações de clusters.

    Parâmetros:
    - df_clusterizado (DataFrame): O DataFrame contendo os dados a serem atualizados.
    - nCluster (list of lists): Uma lista de listas que descrevem os clusters. Cada lista interna
      contém informações sobre o cluster, como cluster ID e tamanho.
    - news_data (list): Uma lista contendo os novos dados a serem incorporados aos clusters.

    Retorna:
    - DataFrame: O DataFrame 'df_clusterizado' atualizado com a coluna 'Media_clients' preenchida
      com base nas informações do 'news_data'. O DataFrame resultante contém apenas as colunas
      'Media_clients' e 'id_client'.

    Descrição:
    Esta função atualiza o DataFrame 'df_clusterizado' preenchendo a coluna 'Media_clients' com base
    nas informações do 'news_data'. Os clusters são identificados com base nas informações em 'nCluster',
    e os dados correspondentes são atribuídos a cada cluster com base nas informações de tamanho.

    A função também atualiza o atributo de classe 'users' com uma lista contendo os valores da coluna
    'Media_clients'.

    Exemplo de uso:
    >>> df = pd.DataFrame({'id_client': [1, 2, 3, 4], 'cluster': [0, 1, 0, 1]})
    >>> nCluster = [[0, 2], 1]
    >>> news_data = [[10, 20, 30, 40]]
    >>> instancia.updated_data(df, nCluster, news_data)
    Retorna o DataFrame atualizado com a coluna 'Media_clients' preenchida.

    """
        selecao = df_clusterizado['cluster'] == nCluster[0][0]
        selecaoo = df_clusterizado.loc[selecao, 'Media_clients']
        dados = news_data[0][:nCluster[0][1]]
        df_clusterizado.loc[selecao, 'Media_clients'] = dados

        selecao = df_clusterizado['cluster'] == nCluster[1]
        selecaoo = df_clusterizado.loc[selecao, 'Media_clients']
        dados = news_data[0][nCluster[0][1]:]
        
        df_clusterizado.loc[selecao, 'Media_clients'] = dados
        self.users = [df_clusterizado['Media_clients'].tolist()]

        return df_clusterizado[['Media_clients', 'id_client']]


#-------------------------------- xxxxxxxxxxxxxxx --------------------------------------#


######################################################################################


    def save_global_model(self):
        """
    Salva o modelo global em um arquivo especificado pelo caminho, considerando o conjunto de dados e o algoritmo.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - global_model: Modelo global a ser salvo.
    - dataset: Nome do conjunto de dados utilizado.
    - algorithm: Nome do algoritmo utilizado para treinamento.

    Retorno:
    - A função não retorna nada explicitamente, mas salva o modelo global em um arquivo.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função save_global_model
    objeto_instancia = MinhaClasse(args, dataset, global_model, algorithm="fedavg")

    # Chamando a função save_global_model para salvar o modelo global
    objeto_instancia.save_global_model()

    Notas Adicionais:
    - Certifique-se de que os atributos global_model, dataset e algorithm estejam devidamente inicializados antes de chamar a função save_global_model.
    - O modelo é salvo em um arquivo no formato PyTorch (.pt) no diretório "models" específico para o conjunto de dados e algoritmo.
        """
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)


    def load_model(self):
        """
    Carrega o modelo global a partir de um arquivo especificado pelo caminho, considerando o conjunto de dados e o algoritmo.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - global_model: Modelo global a ser carregado.
    - dataset: Nome do conjunto de dados utilizado.
    - algorithm: Nome do algoritmo utilizado para treinamento.

    Retorno:
    - A função não retorna nada explicitamente, mas atualiza o modelo global com o modelo carregado.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função load_model
    objeto_instancia = MinhaClasse(args, dataset, global_model, algorithm="fedavg")

    # Chamando a função load_model para carregar o modelo global
    objeto_instancia.load_model()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos global_model, dataset e algorithm estejam devidamente inicializados antes de chamar a função load_model.
    - O modelo é carregado a partir de um arquivo no formato PyTorch (.pt) no diretório "models" específico para o conjunto de dados e algoritmo.
    ```
        """
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)


    def model_exists(self):
        """
    Verifica se um modelo global existe no diretório especificado pelo caminho, considerando o conjunto de dados e o algoritmo.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - dataset: Nome do conjunto de dados utilizado.
    - algorithm: Nome do algoritmo utilizado para treinamento.

    Retorno:
    - bool: True se um modelo global existe, False caso contrário.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função model_exists
    objeto_instancia = MinhaClasse(args, dataset, algorithm="fedavg")

    # Chamando a função model_exists para verificar a existência do modelo global
    modelo_existe = objeto_instancia.model_exists()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos dataset e algorithm estejam devidamente inicializados antes de chamar a função model_exists.
    - A função retorna True se um modelo global existir no diretório específico para o conjunto de dados e algoritmo, e False caso contrário.
        """
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)


    def save_results(self):
        """
    Salva os resultados de avaliação em um arquivo HDF5 no diretório especificado, considerando o conjunto de dados, algoritmo e métricas.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - dataset: Nome do conjunto de dados utilizado.
    - algorithm: Nome do algoritmo utilizado para treinamento.
    - goal: Objetivo da avaliação (opcional).
    - times: Número de vezes que a avaliação foi realizada (opcional).
    - rs_test_acc: Lista de acurácias nos testes.
    - rs_test_auc: Lista de áreas sob a curva ROC nos testes.
    - rs_train_loss: Lista de perdas nos treinamentos.

    Retorno:
    - A função não retorna nada explicitamente, mas salva os resultados em um arquivo HDF5.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função save_results
    objeto_instancia = MinhaClasse(args, dataset, algorithm="fedavg", rs_test_acc=[0.85, 0.88], rs_test_auc=[0.92, 0.94], rs_train_loss=[0.2, 0.15])

    # Chamando a função save_results para salvar os resultados de avaliação
    objeto_instancia.save_results()
        

    Notas Adicionais:
    - Certifique-se de que os atributos dataset, algorithm e as listas de resultados estejam devidamente inicializados antes de chamar a função save_results.
    - Os resultados são salvos em um arquivo HDF5 no diretório "../results/" com um nome formatado de acordo com o conjunto de dados, algoritmo e, opcionalmente, objetivo e número de vezes.
        """
        
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
        """
    Salva um item (por exemplo, um modelo ou um tensor) em um arquivo PyTorch no diretório especificado.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - item: Item a ser salvo, como um modelo PyTorch ou um tensor.
    - item_name: Nome associado ao item para compor o nome do arquivo de salvamento.

    Atributos Esperados em self:
    - save_folder_name: Caminho do diretório onde o item será salvo.

    Retorno:
    - A função não retorna nada explicitamente, mas salva o item em um arquivo PyTorch.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função save_item
    objeto_instancia = MinhaClasse(args, save_folder_name="saved_items")

    # Chamando a função save_item para salvar um modelo chamado 'meu_modelo'
    objeto_instancia.save_item(meu_modelo, "meu_modelo")
    ```

    Notas Adicionais:
    - Certifique-se de que o atributo save_folder_name esteja devidamente inicializado antes de chamar a função save_item.
    - O item é salvo em um arquivo PyTorch no diretório especificado, com um nome formatado de acordo com o prefixo "server_" e o item_name.
        """

        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def load_item(self, item_name):
        """
    Carrega um item (por exemplo, um modelo ou um tensor) a partir de um arquivo PyTorch no diretório especificado.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - item_name: Nome associado ao item para compor o nome do arquivo de carregamento.

    Atributos Esperados em self:
    - save_folder_name: Caminho do diretório onde o item está salvo.

    Retorno:
    - item: Item carregado, como um modelo PyTorch ou um tensor.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função load_item
    objeto_instancia = MinhaClasse(args, save_folder_name="saved_items")

    # Chamando a função load_item para carregar um modelo chamado 'meu_modelo'
    meu_modelo = objeto_instancia.load_item("meu_modelo")
    ```

    Notas Adicionais:
    - Certifique-se de que o atributo save_folder_name esteja devidamente inicializado antes de chamar a função load_item.
    - O item é carregado a partir de um arquivo PyTorch no diretório especificado, com um nome formatado de acordo com o prefixo "server_" e o item_name.
        """

        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def test_metrics(self):
        """
    Avalia as métricas de desempenho do modelo global nos clientes participantes.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - eval_new_clients: Booleano indicando se a avaliação inclui novos clientes.
    - num_new_clients: Número de novos clientes, se a avaliação incluir novos clientes.
    - clients: Lista de clientes participantes.
    
    Retorno:
    - Se eval_new_clients for True e num_new_clients for maior que 0, a função realiza ajuste fino nos novos clientes e retorna os resultados de test_metrics_new_clients.
    - Caso contrário, retorna listas contendo IDs dos clientes, número de amostras testadas, número total de classificações corretas e área sob a curva ROC (AUC) ponderada.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função test_metrics
    objeto_instancia = MinhaClasse(args, clients=lista_de_clientes, eval_new_clients=True, num_new_clients=5)

    # Chamando a função test_metrics para avaliar as métricas de desempenho
    resultado_metrics = objeto_instancia.test_metrics()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos eval_new_clients, num_new_clients e clients estejam devidamente inicializados antes de chamar a função test_metrics.
    - Se eval_new_clients for True e num_new_clients for maior que 0, a função realiza ajuste fino nos novos clientes antes da avaliação.
    - Os resultados incluem listas contendo IDs dos clientes, número de amostras testadas, número total de classificações corretas e AUC ponderada.
        """

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
        """
    Avalia as métricas de desempenho durante o treinamento nos clientes participantes.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - eval_new_clients: Booleano indicando se a avaliação inclui novos clientes.
    - num_new_clients: Número de novos clientes, se a avaliação incluir novos clientes.
    - clients: Lista de clientes participantes.
    
    Retorno:
    - Se eval_new_clients for True e num_new_clients for maior que 0, a função retorna listas fictícias [0], [1], [0].
    - Caso contrário, retorna listas contendo IDs dos clientes, número de amostras utilizadas no treinamento e perda (loss) ponderada.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função train_metrics
    objeto_instancia = MinhaClasse(args, clients=lista_de_clientes, eval_new_clients=True, num_new_clients=5)

    # Chamando a função train_metrics para avaliar as métricas de desempenho durante o treinamento
    resultado_metrics = objeto_instancia.train_metrics()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos eval_new_clients, num_new_clients e clients estejam devidamente inicializados antes de chamar a função train_metrics.
    - Se eval_new_clients for True e num_new_clients for maior que 0, a função retorna listas fictícias para indicar que a avaliação não é aplicável durante o treinamento.
    - Os resultados incluem listas contendo IDs dos clientes, número de amostras utilizadas no treinamento e a perda ponderada.
        """
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
        """
    Avalia o desempenho do modelo global utilizando métricas de teste e treinamento.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - acc: Lista opcional para armazenar valores de acurácia (pode ser None).
    - loss: Lista opcional para armazenar valores de perda (pode ser None).

    Atributos Esperados em self:
    - rs_test_acc: Lista para armazenar acurácias médias nos testes em diferentes rodadas.
    - rs_train_loss: Lista para armazenar perdas médias nos treinamentos em diferentes rodadas.

    Retorno:
    - A função não retorna nada explicitamente, mas atualiza as listas rs_test_acc e rs_train_loss com os resultados da avaliação.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função evaluate
    objeto_instancia = MinhaClasse(args, rs_test_acc=[], rs_train_loss=[])

    # Chamando a função evaluate para avaliar o desempenho do modelo global
    objeto_instancia.evaluate()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos rs_test_acc e rs_train_loss estejam devidamente inicializados antes de chamar a função evaluate.
    - Os resultados da avaliação são impressos no console e escritos no arquivo "saida.txt".
    - Se acc e loss forem fornecidos, os resultados da avaliação são adicionados às listas correspondentes, caso contrário, são adicionados às listas rs_test_acc e rs_train_loss.
        """

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        with open("saida.txt", "a") as arquivo:
        # Escrever o valor da variável no arquivo
            arquivo.write(str(test_acc) + "," + str(train_loss)+ "\n")
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
        """
    Imprime métricas de desempenho na console.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - test_acc: Acurácia média nos testes.
    - test_auc: Área sob a curva ROC média nos testes.
    - train_loss: Perda média nos treinamentos.

    Retorno:
    - A função não retorna nada explicitamente, apenas imprime as métricas no console.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função print_
    objeto_instancia = MinhaClasse(args)

    # Chamando a função print_ para imprimir métricas de desempenho
    objeto_instancia.print_(0.85, 0.92, 0.2)
    ```

    Notas Adicionais:
    - Certifique-se de que os valores de test_acc, test_auc e train_loss estejam disponíveis antes de chamar a função print_.
    - Os resultados são impressos no console em formato legível.
        """
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))


    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        """
    Verifica se determinados critérios foram atendidos com base em listas de acurácias ou perdas.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - acc_lss: Lista de listas contendo acurácias ou perdas ao longo de diferentes rodadas ou iterações.
    - top_cnt: Número máximo permitido de valores no topo para considerar como convergência (opcional).
    - div_value: Valor máximo permitido para o desvio padrão das últimas top_cnt acurácias ou perdas (opcional).

    Retorno:
    - bool: True se os critérios especificados foram atendidos para todas as listas, False caso contrário.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função check_done
    objeto_instancia = MinhaClasse(args)

    # Verificando se os critérios foram atendidos para a lista de acurácias acc_lss
    criterios_atendidos = objeto_instancia.check_done(acc_lss, top_cnt=3, div_value=0.05)
    ```

    Notas Adicionais:
    - Certifique-se de que os valores top_cnt e div_value estejam corretamente especificados antes de chamar a função check_done.
    - Se top_cnt e div_value forem fornecidos, a função verifica se a diferença entre o maior valor e o segundo maior é maior que top_cnt e se o desvio padrão das últimas top_cnt acurácias ou perdas é menor que div_value.
    - Se apenas top_cnt ou div_value for fornecido, a função verifica apenas o critério correspondente.
    - A função retorna True se os critérios forem atendidos para todas as listas, caso contrário, retorna False.
        """

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
        """
    Chama a função DLG (Distributed Learning with Gradients) para avaliação do desempenho do modelo global em relação aos gradientes locais dos clientes.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - R: Parâmetro específico a ser passado para a função DLG.

    Atributos Esperados em self:
    - uploaded_ids: Lista de IDs dos clientes que enviaram seus modelos.
    - uploaded_models: Lista de modelos enviados pelos clientes.
    - global_model: Modelo global treinado.
    - batch_num_per_client: Número máximo de lotes por cliente para avaliação.

    Retorno:
    - A função não retorna nada explicitamente, mas imprime o valor da PSNR (Peak Signal-to-Noise Ratio) média em decibéis.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função call_dlg
    objeto_instancia = MinhaClasse(args)

    # Chamando a função call_dlg para avaliar o desempenho do modelo global com DLG
    objeto_instancia.call_dlg(R=0.5)
    ```

    Notas Adicionais:
    - Certifique-se de que os modelos e IDs dos clientes estejam disponíveis antes de chamar a função call_dlg.
    - A função avalia o desempenho do modelo global em relação aos gradientes locais dos clientes usando a função DLG.
    - O resultado, a PSNR média, é impresso no console.
        """

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
        """
    Adiciona novos clientes à lista de novos clientes.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.
    - clientObj: Classe que define a estrutura dos novos clientes.

    Atributos Esperados em self:
    - num_clients: Número atual de clientes no sistema.
    - num_new_clients: Número de novos clientes a serem adicionados.
    - dataset: Nome do conjunto de dados sendo utilizado.
    - args: Argumentos necessários para inicializar um novo cliente.
    - new_clients: Lista de novos clientes a ser preenchida.

    Retorno:
    - A função não retorna nada explicitamente, mas adiciona novos clientes à lista new_clients.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função set_new_clients
    objeto_instancia = MinhaClasse(args, num_clients=10, num_new_clients=5, dataset="mnist")

    # Chamando a função set_new_clients para adicionar novos clientes à lista
    objeto_instancia.set_new_clients(clientObj=NovoCliente)
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos num_clients, num_new_clients, dataset e args estejam devidamente inicializados antes de chamar a função set_new_clients.
    - A função cria novos clientes usando a classe clientObj e os adiciona à lista new_clients.
        """

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
        """
    Realiza ajuste fino nos novos clientes utilizando o modelo global como ponto de partida.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - new_clients: Lista de novos clientes que passarão por ajuste fino.
    - global_model: Modelo global utilizado como ponto de partida.
    - learning_rate: Taxa de aprendizado para o otimizador durante o ajuste fino.
    - fine_tuning_epoch: Número de épocas para o ajuste fino.

    Retorno:
    - A função não retorna nada explicitamente, mas realiza o ajuste fino nos modelos dos novos clientes.

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função fine_tuning_new_clients
    objeto_instancia = MinhaClasse(args, new_clients=lista_de_novos_clientes, global_model=modelo_global)

    # Chamando a função fine_tuning_new_clients para realizar ajuste fino nos novos clientes
    objeto_instancia.fine_tuning_new_clients()
    ```

    Notas Adicionais:
    - Certifique-se de que os atributos new_clients, global_model, learning_rate e fine_tuning_epoch estejam devidamente inicializados antes de chamar a função fine_tuning_new_clients.
    - A função utiliza o otimizador SGD e a função de perda CrossEntropyLoss para realizar o ajuste fino nos novos clientes.
        """


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
        """
    Avalia as métricas de desempenho nos novos clientes.

    Parâmetros:
    - self: O objeto que invoca a função, uma instância de uma classe contendo atributos necessários.

    Atributos Esperados em self:
    - new_clients: Lista de novos clientes a serem avaliados.

    Retorno:
    - Tuple: Retorna uma tupla contendo IDs dos novos clientes, número de amostras utilizadas nos testes, total de classificações corretas e total da área sob a curva ROC (AUC).

    Exemplo de Uso:
    ```python
    # Criando uma instância da classe que contém a função test_metrics_new_clients
    objeto_instancia = MinhaClasse(args, new_clients=lista_de_novos_clientes)

    # Chamando a função test_metrics_new_clients para avaliar as métricas de desempenho nos novos clientes
    resultados_metrics = objeto_instancia.test_metrics_new_clients()
    ```

    Notas Adicionais:
    - Certifique-se de que a lista de novos clientes (new_clients) esteja devidamente inicializada antes de chamar a função test_metrics_new_clients.
    - Os resultados incluem IDs dos novos clientes, número de amostras utilizadas nos testes, total de classificações corretas e total da área sob a curva ROC (AUC).
        """
        
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
