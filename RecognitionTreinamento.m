
% Autor: Rodrigo Leal

clear; clc;close all;

%Dados de entrada e o desejado, criando o gr?fico inicial.

%Cria 1000 valores de entrada
Entrada = [1 2 2 2 1; 1 2 2 1 1; 1 2 1 1 1; 2 2 2 1 1; 1 1 1 1 1];

% Função de Saída
Desejado = [1 2 3 4 5];

disp (Entrada)
disp (Desejado)

%Quantidades de neurônios na camada de Entrada In
In = 5;

%Quantidade de neurônios na camada Escondida H
H = 8;

%Quantidade de neurônios na camada de Saida Out
Out = 1;

%Definindo a taxa de aprendizagem - Valor Eta.
eta = 0.00022;

%Quantidade de Épocas
Epocas = 5000;

%Se a fun??o de ativação é linear, então Defini-se o valor da constante k.
k = 1; 

% Matriz que imprime o gráfico do erro quadrático médio
grafico_erro = [];

% Algoritmo

% Iniciando os Pesos - Primeiro Teste - logo apps a primeira inicialização,
% comentar código.
    
    Whi = rand(H,In) - 0.5;
    Woh = rand(Out,H) - 0.5;
    
%load pesos15 Woh Whi

for In=0:Epocas
    
    %Calculo da entrada da camada escondida.
    net_h = Whi*Entrada;
     
    %Calculo da sa?da da camada escondida - aplicar funçãoo de ativação.
    Output_h = logsig(net_h);
             
    %Calculo da entrada da camada de sa?da. 
    net_o = Woh*Output_h;
  
    %Calcular a sa?da da camada de sa?da (Sa?da da RNA).
    
    %calculo do valor da sa?da:
    Output = k*net_o;
         
    %Calcular o erro da sa?da.
    Erro = Desejado - Output;

            %Backpropagation para recalcular os pesos, calculando a variação 
            %dos pesos entre Woh.
           
            %C?lculo da derivada.
            df = k*ones(size(net_o));
                             
            %C?lculo do delta Woh, valor da variação.
            delta_Woh = eta*(Erro.*df)*Output_h';

            %C?lculo do erro retropopagado.
            Erro_r = Woh'*(Erro.*df);
        
            %C?lculo da varia??o dos pesos Whi.
            %C?lculo da Derivada.
            df = logsig(net_h)-(logsig(net_h).^2);
            
            %C?lculo do delta Whi.
            delta_Whi = eta*(Erro_r.*df)*Entrada';
    
            %C?lculo dos novos pesos.
            Whi = Whi+delta_Whi;
            Woh = Woh+delta_Woh;
        
            %C?lculo do Erro Quadrático M?dio.
            emq = sqrt(sum(Erro.^2))/size(Erro,2);
            grafico_erro = [grafico_erro emq];
        
end

%Imprime o gr?fico referente ao Erro Quadr?tico M?dio.
figure(1);
plot(grafico_erro, 'r');
hold on;
grid on;
disp(In);
xlabel('Épocas');
ylabel('Erro (EQM)');
legend('Erro Quadrático Médio');

%Grava os pesos atuais para posteriores opera??es.
save pesos2 Woh Whi

%Grava pesos para serem utilizados no teste de generaliza??o
save pesoteste Woh Whi