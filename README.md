# "Aumente o Faturamento da Empresa com Previsão de Churn"

Aqui eu deixo uma versão mais amigável do projeto, onde esponho de forma resumida cada fasse do notebook original, dando foco na parte de negócio e resolução, se trata de um projeto-desafio que pode ser encontrado no blog "Seja um Data Scientist". 

Nesse link: https://www.kaggle.com/mervetorkan/churndataset

Nesse desafio uma empresa fictícia fornece uma base de dados se queixando dos últimos meses, onde muitos clientes vieram a cancelar suas contas diminuindo assim o faturamento da empresa, na base de dados temos as informações:

CustomerID: Identificador do Cliente
Surname: Sobrenome do cliente.
CreditScore: A pontuação de Crédito do cliente para o mercado de consumo. Equivalente ao nosso "Score"
Geography: O país onde o cliente reside.
Gender: O gênero do cliente.
Age: A idade do cliente.
Tenure: Número de anos que o cliente permaneceu ou está ativo no banco.
Balance: Valor monetário que o cliente tem em sua conta bancária.
NumOfProducts: O número de produtos comprado pelo cliente no banco.
HasCrCard: Indica se o cliente possui ou não cartão de crédito.
IsActiveMember: Indica se o cliente fez pelo menos uma movimentação na conta bancário dentro de 12 meses.
EstimateSalary: Estimativa do salário mensal do cliente.
Exited: Indica se o cliente saiu ou não do banco.

  Sendo "1" sim e "0" não.

## O problema:

O número de clientes perdidos nos últimos períodos veio em patamares record, como cientista de dados devo verificar a causa do problema e tranformar a solução em uma solução de DataScience.

A solução de DataScience: Criar um modelo que liste com antecedência os clientes que podem vir a cancelar suas contas no banco

# Minha resolução:

## Checar Tipos e Nulos:

Não havia dados nulos para serem preenchidos, quantoa  tipagem dos dados, a coluna de "CustomerID" foi lida como inteiro pelo pandas mas se trata de um objeto, portanto foi a única correção a ser feita nessa parte
![image](https://user-images.githubusercontent.com/85971408/129573111-ed928a4b-c1b3-4fd7-a117-3b5bf0d60ff0.png)

## Ver o balanço das classes:

Antes mesmo de ir para a análise exploratória eu gosto de observar como está o balanço da base, isso trás mais inteligência para os insights.
Além disso, classes desbalanceadas podem enviesar bastante a percepção de acertividade de um modelo;
![image](https://user-images.githubusercontent.com/85971408/129573424-30fbb893-04e7-4cf3-8ab6-13bc2aee1ae8.png)

Com isso é possível ver que a base é desbalanceada, com mais clientes pertencentes a classe "0" que foi a classe que ficou no banco.

## Checagem de Outliers:

Aqui não houve muita surpresa, a única classe que me chamou a atenção foi a de balanços, alguns clientes tinham balanços bastante grandes
![image](https://user-images.githubusercontent.com/85971408/129573957-c0b9d14f-4cf3-4005-9832-5205b7bc7246.png)

Algo importante para ser reparado foi que os "top 5" balanços do banco estraram em "churn", portanto isso é um indicativo incial de cuidado com essa categoria de clientes.

## EDA:

A primeira variável que quero destacar nessa fasse é a idade, como é possível ver pelas distribuições a classe de cima "1" é bem mais velha que a classe de baixo "0".

Portanto conclue-se que clientes mais velhos estão deixando mais o banco
![image](https://user-images.githubusercontent.com/85971408/129574296-68b7a0c4-b42d-4c04-89de-28e940acae20.png)

Outro fator importante à ser elencado também é o "CreditScore" onde a distribuição de Scores na classe que saiu é mais "homogênia"

![image](https://user-images.githubusercontent.com/85971408/129578167-7a60fcc0-63ee-4979-8a11-ca579edb497d.png)

No atributo "IsActiveMember" é possível notar que os membros inavativos sairám muito mais em termos percentuais que os inativos
![image](https://user-images.githubusercontent.com/85971408/129578549-a68d599b-cc82-4f7a-aeee-b9027645c89f.png)

Nos países do Dataset, há um maior nível de Churn na Alemanha:

![image](https://user-images.githubusercontent.com/85971408/129578806-4a32436f-eb62-4694-9201-27a65b2a4923.png)
Onde praticamente metade dos clientes foram perdidos.

### **Conclusão Geral da EDA:**

- **Pessoas mais velhas cancelaram mais suas contas que pessoas mais novas.**

- **Clientes Inativos a mais de um ano cancelaram mais suas contas.**

- **Clientes com score mais baixo cancelaram mais suas contas.**

- **A Alemanha teve o pior nível de churn.**

- **A base está bem desbalanceada, necessário se ater a isso para medir correlação apenas após o balanceamento, e jogar os dados no modelo depois.**

# Algoritimos de ML:

obs: Antes de rodar os modelos foi feito:

- Escalonamento dos valores.
- Definição de treino & teste.
- Tranformação de categporicos em numéricos.
- Separação de Treino & Teste.
- Estudado Correlação entre variáveis.
- Treino & Teste salvos como arquivo Pickle.

## Avore de decisão

Gosto de rodar esse algoritimo primeiro pois consigo além de ter um resultado incial observar as "features importances" oque trás ganho de informação para os próximos algoritimos
![image](https://user-images.githubusercontent.com/85971408/129580317-cba1b1fc-362c-4891-ba4e-29fa37596d20.png)

O resultado inicial foi bastante ruim, tendo em vista que a precisão para a classe "0" está bem melhor

### Feature Importances:

![image](https://user-images.githubusercontent.com/85971408/129580653-6e6fa380-d8fd-4bc1-b9f4-71efd7cf9327.png)

Nesse gráfico podemos observar qua as variáveis com mais ganho de informação são a Idade, Salário, Balanço, Score, Número de Produtos, e Quantia de anos ativo.

##SVM
![image](https://user-images.githubusercontent.com/85971408/129584996-9671a4eb-4812-4ff5-9618-f3dbda612cba.png)

Apesar do bom reasultado geral, o eesultado ainda deixa a desejar para a classe "1".

##Rede Neural
![image](https://user-images.githubusercontent.com/85971408/129585422-5f3fb643-0564-4710-accf-fafffdac600e.png)

Apesar do bom reasultado geral, o eesultado ainda deixa a desejar para a classe "1".

# Tunning:

Aqui é feito uma listagem dos parâmetros possíveis para os algorítimos e são testados diferentes combinações de parâmetros, os melhores resultados para cada vieram dos parâmetros:

**Arvore:{'criterion': 'gini', 'min_samples_leaf': 10, 'min_samples_split': 15, 'splitter': 'random'}**

**SVM:{'C': 2.0, 'kernel': 'rbf'}**
obs:gostariade ter testado mais parâmetros no SVM, porém o Colab estava derrubando devido a demora no processo de tunning

**Rede Neural:{'activation': 'logistic', 'batch_size': 10, 'solver': 'adam'}**

# Validação Cruzada:

Aqui são testados os algoritimos com os padrões melhorados, e com várias combinações diferentes de treino e teste, depois os resultados são colocados em um DataFrame para serem avaliados comparativamente

![image](https://user-images.githubusercontent.com/85971408/129588261-d1e9851a-ef63-4b61-9b5f-afcceca92106.png)

# Teste De normalidade:
Para verificar se os modelos tem resultados próximos, é feito uma validação estatística das distribuições dos resultados, porém esse teste só funciona quando o intervalo de resultados é uma distribuição normal, aqui, antes dos testes comparativos, verificamos se as distribuições de resultados são normais.
svm
![image](https://user-images.githubusercontent.com/85971408/129588358-55e35b57-646f-4c62-a7fa-b90a2bdc5036.png)
arvore
![image](https://user-images.githubusercontent.com/85971408/129588424-ebba9b6a-d1ef-4666-a185-dc5103f6a85b.png)
rede
![image](https://user-images.githubusercontent.com/85971408/129588465-13038d8f-5cdc-4282-bb6a-0a8f2efc0467.png)

# Teste de hipótese:
Agora verificamos se entre os algoritimos há uma diferença significativa de performance
![image](https://user-images.githubusercontent.com/85971408/129588662-69332cef-1fda-4b5f-adfb-95bbc1d29ad8.png)
![image](https://user-images.githubusercontent.com/85971408/129588697-67c4e7cf-0df3-47ef-b470-0c5ee6cfd797.png)
![image](https://user-images.githubusercontent.com/85971408/129588722-87670c1b-6cd4-4f80-b98f-dfe17915037d.png)

Pela avalização de Tukey apenas a árvore de decisão teve um resultado suficientemente ruim a ponto de ser considera inferior

# Salvar os parâmetros:

![image](https://user-images.githubusercontent.com/85971408/129590118-5bc95dd6-4bf4-4fbd-a8ce-94e77292e3b8.png)
