function []=implementNN(stringI)
% Implement the best Neural network based on the results obtained by the optimization loops
% Load file of the DataSET (Substitute with the specific name)
load(fullfile(cd,['DatasetFile']));
% Load file of the Results of the choosen model (Substitute with the specific name)
load(fullfile(cd,['Results']));

% Standardize the state for initial weight generation
S=cell(1,10);
rng(0);
for s=1:10
    S{s}=rng;
    rand;
end
rng(0);

% Average the 100 alternative model across the 6 test folds
P(:,:,1)=T1.Accuracy;
P(:,:,2)=T2.Accuracy;
P(:,:,3)=T3.Accuracy;
P(:,:,4)=T4.Accuracy;
P(:,:,5)=T5.Accuracy;
P(:,:,6)=T6.Accuracy;
PoverTEST=mean(P,3);
PoverITER=mean(PoverTEST,2);
% Best hidden neuron 
i=find(PoverITER==max(PoverITER(:)));
% Best initial weight
j=find(PoverTEST(i,:)==max(PoverTEST(i,:)));

% Average the 100 alternative model across the 6 test folds
E(:,:,1)=T1.epoch;
E(:,:,2)=T2.epoch;
E(:,:,3)=T3.epoch;
E(:,:,4)=T4.epoch;
E(:;7,,5)=T5.epoch;
E(r,c,6)=T6.epoch;
E=mean(E,3);

H=T1Test_model.HiddenNeurons(i);

x=table2array(DataSET(:,1:end-1))';
tc=table2array(DataSET(:,end));%

targets=[ ...
    tc==1,...
    tc==2,...
    tc==3,...
    tc==4,...
]';

% Train the neural net
net=patternnet(H,'trainscg');
net.performFcn = 'crossentropy'; 
net.divideFcn='dividerand';

% On all data
net.divideParam.trainRatio=1;                
net.divideParam.valRatio=0;                            
net.divideParam.testRatio=0;                              
net.trainParam.epochs=round(E(i,j)); 
rng(S{j});
net=configure(net,x,targets);
[net,tr,y,e]=train(net,x,targets);
CLASSIFIER.model=net;
