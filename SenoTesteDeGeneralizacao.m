%Dados de entrada e o desejado, criando o gráfico inicial.

%Cria 100 valores de entrada
%EntradaTeste = -pi:pi/499.8:pi;
EntradaTeste = rand(1, 100)*2*pi-pi;

Desejado = sin(EntradaTeste);

disp (EntradaTeste)
disp (Desejado)

%Treinamento da Rede Neural Artificial (RNA) para resolução da função
%Seno.

%Quantidades de neurônios na camada de Entrada In
In = 1;

%Quantidade de neurônios na camada Escondida H
H = 4;

%Quantidade de neurônios na camada de Saida Out
Out = 1; 

%Definindo a taxa de aprendizagem - Valor Eta.
eta = 0.000002;

%Se a função de ativação é linear, então Defini-se o valor da constante k.
k = 1; 

%Algoritmo
    
load pesoteste Woh Whi

%Carrega o valor dos dados de teste

    
    %Calculo da entrada da camada escondida.
    net_h = Whi*EntradaTeste;
     
    %Calculo da saída da camada escondida - aplicar função de ativação.
    Output_h = logsig(net_h);
             
    %Calculo da entrada da camada de saída. 
    net_o = Woh*Output_h;
  
    %Calcular a saída da camada de saída (Saída da RNA).
    
    %calculo do valor da saída:
    Output = k*net_o;
         
    %Calcular o erro da saída.
    Erro = Desejado - Output;