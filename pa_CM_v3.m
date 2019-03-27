close all
clear all
clc

I_s=0.01e-12;
I_b=0.1e-12;
V_b=1.3;
G_p=0.1;


current= @(V) I_s.*(exp(1.2/0.025.*V)-1)+ G_p.*V-I_b.*(exp(-1.2/0.025.*(V+V_b)-1));

V=linspace(-1.95,0.7,200);
I_1=current(V);
I_N=I_1+I_1.*(0.4*rand(1,length(I_1))-0.2);

inputs = V;
targets = I_N;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs

%% Plot graphs
figure(7)
hold on;
plot(V,I_N);
plot(V,Inn);
hold off;
legend('Raw','nn');


figure(8)
semilogy(V,abs(I_N));
hold on;
semilogy(V,abs(Inn));
hold off;
legend('Raw','nn');

