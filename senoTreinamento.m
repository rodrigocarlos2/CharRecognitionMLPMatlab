%Autor: Filipe Fontinele - Tabalho baseado em redes neurais MLP com treinamento BP
%exemplificado com a função seno.

%limpa todos os valores passados ao matlab, necessário para evitar
%possíveis interferências de valores indesejados.
clear; clc;close all;

%Dados de entrada e o desejado, criando o gráfico inicial.

%Cria 1000 valores de entrada
Entrada = -pi:pi/499.8:pi;

Desejado = sin(Entrada);

disp (Entrada)
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

%Quantidade de épocas
Epocas = 4000; 

%Se a função de ativação é linear, então Defini-se o valor da constante k.
k = 1; 

%Matriz que imprime o gráfico do erro quadrátido médio
grafico_erro = [];

%Algoritmo

%Iniciando os Pesos - Primeiro Teste - logo após a primeira inicialização,
%comentar código.
    
    Whi = rand(H,In) - 0.5;
    Woh = rand(Out,H) - 0.5;
    
%load pesos1 Woh Whi

for In=0:Epocas
    
    %Calculo da entrada da camada escondida.
    net_h = Whi*Entrada;
     
    %Calculo da saída da camada escondida - aplicar função de ativação.
    Output_h = logsig(net_h);
             
    %Calculo da entrada da camada de saída. 
    net_o = Woh*Output_h;
  
    %Calcular a saída da camada de saída (Saída da RNA).
    
    %calculo do valor da saída:
    Output = k*net_o;
         
    %Calcular o erro da saída.
    Erro = Desejado - Output;

            %Backpropagation para recalcular os pesos, calculando a variação 
            %dos pesos entre Woh.
           
            %Cálculo da derivada.
            df = k*ones(size(net_o));
                             
            %Cálculo do delta Woh, valor da variação.
            delta_Woh = eta*(Erro.*df)*Output_h';

            %Cálculo do erro retropopagado.
            Erro_r = Woh'*(Erro.*df);
        
            %Cálculo da variação dos pesos Whi.
            %Cálculo da Derivada.
            df = logsig(net_h)-(logsig(net_h).^2);
            
            %Cálculo do delta Whi.
            delta_Whi = eta*(Erro_r.*df)*Entrada';
    
            %Cálculo dos novos pesos.
            Whi = Whi+delta_Whi;
            Woh = Woh+delta_Woh;
        
            %Cálculo do Erro Quadrático Médio.
            emq = sqrt(sum(Erro.^2))/size(Erro,2);
            grafico_erro = [grafico_erro emq];
        
end

%Imprime o gráfico referente ao seno esperado e o seno obtido.
figure(1);
plot(Entrada, Output,'g', Entrada, Desejado,'black');
hold on;
grid on;
title('Seno esperado (Preto) e o Seno obtido (Verde)');

%Imprime o gráfico referente ao Erro Quadrático Médio.
figure(2);
plot(grafico_erro, 'r');
hold on;
grid on;
disp(In);
xlabel('Épocas');
ylabel('Erro (EQM)');
legend('Erro Quadrático Médio');

%Grava os pesos atuais para posteriores operações.
save pesos1 Woh Whi

%Grava pesos para serem utilizados no teste de generalização
save pesoteste Woh Whi