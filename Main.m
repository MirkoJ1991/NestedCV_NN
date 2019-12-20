% Clear memory
clear all; close all; clc

%% LOAD DATASET
% Load of the DataSET: Observation X Features + Response
load(fullfile(''));

% Inputs
x=table2array(DataSET(:,1:end-1))';

% Targets
tc=table2array(DataSET(:,end));
targets=[tc==1,tc==2,tc==3,tc==4]';

% Dimensionality of the dataset
[I N]=size(x);
[O ~]=size(targets);

%% DEFINITION OF DATASETS (TRAINING/VALIDATION/TEST)
% Divisor contains subject's labels for each observation
Sbj=unique(DIVISOR);
loop=0;
while loop==0
    flag=input(['What validation model you would like to implement?\n   1 - 5 folds\n   2 - 10 folds\n   3 - LOSOCV\n\n']);
    switch flag
        case 1
            % 6 x 5 CV
            folds = 6;
            loop = 1;           
        case 2
            % 11 x 10 CV
            folds = 11;
            loop = 1;
        case 3
            % LOSOCV (Leave One Subject Out CrossValidation)
            folds = length(SBJ);
            loop = 1;
        otherwise
            loop = 0;
    end
end

%% FOLDER PREPARATION
% Randomize Subjects
Shuffled=Sbj(randperm(numel(Sbj)));

% Min number of subjects in a fold
step=floor(numel(Sbj)/folds);

% Index to shuffle subjects inside folds
division=1:step:(step*folds);
% If R is not zero the number of subjects can not equally be distribute to
% each fold 
R=rem(numel(Sbj),folds);

% Fill the folds
subgroup=cell(folds,1);
for t=1:folds
    subgroup{t}=Shuffled(division(t):division(t)+step-1);
end
if R~=0
    % Remaining subjects
    Remain=Shuffled(end-(R-1):end);
    subgroup=Redistribute(DIVISOR,subgroup,Remain);%------------------------- Ridistribuzione
end

% The number of neurons are between INPUT and OUTPUT
dn=floor((I-O)/15);
Neurons=((0:9)*dn)+4;

%% NEURAL NETS IMPLEMENTATION

for t=1:folds
    % For each fold 
    % I identify the Test fold
    logicalindext=cellfun(@(x)contains(DIVISOR,x),subgroup{t},'un',0);                          
    ITST=find(any(horzcat(logicalindext{:}),2)==1);
    % And the validation/training folds
    IVAL=cell(1,folds-1);
    ITRN=cell(1,folds-1);
    count=1;  
    for v=1:folds
        if t~=v
            logicalindexv=cellfun(@(x)contains(DIVISOR,x),subgroup{v},'un',0);
            IVAL{1,count}=find(any(horzcat(logicalindexv{:}),2)==1);
            ITRN{1,count}=find(~any([any(horzcat(logicalindext{:}),2),any(horzcat(logicalindexv{:}),2)],2)==1);
            count=count+1;
        end
    end
   
    % Random state for initial weight
    S=cell(1,folds-1);
    rng(0);
    for s=1:10
        S{s}=rng;
        rand;
    end
    rng(0);
    
    % Declaration of data structure
    perf_xentrval=cell(10,10); perf_xentrtrn=cell(10,10); perf_xentrtst=zeros(10,10);
    perf_mseval=cell(10,10); perf_msetrn=cell(10,10); perf_msetst=zeros(10,10);
    perf_best_ep=cell(10,10); perf_TSToutputs=cell(10,10); perf_TSTtargets=cell(10,10);
    accuracy=zeros(10,10); recall=zeros(10,10); precision=zeros(10,10); f1=zeros(10,10);
    % For each model defined by the hidden number of neurons
    for n=1:10
        % Number of neurons
        H=Neurons(n);
        parfor i=1:10
            % Start a new cicle of validation over the remaining folds aside from the test fold          
            fprintf(['Validation for Model with: ',num2str(H),' neurons and randomization ',num2str(i),'\n']);
            
            % ----> ValidationLoops
            [val_xentrval,val_xentrtrn,val_mseval,val_msetrn,val_R2,val_R2trn,val_R2val,val_epochs]=ValidationLoops(S{i},MSE00,MSE00trn,MSE00val,folds,x,targets,H,ITRN,IVAL,ITST)
            
            % Save model parameters
            perf_xentrval{n,i}=[mean(val_xentrval),std(val_xentrval)];
            perf_xentrtrn{n,i}=[mean(val_xentrtrn),std(val_xentrtrn)];
            perf_mseval{n,i}=[mean(val_mseval),std(val_mseval)];
            perf_msetrn{n,i}=[mean(val_msetrn),std(val_msetrn)];
            perf_best_ep{n,i}=[mean(val_epochs),std(val_epochs)];
            
            % Initialization of the optimal net to be tested on the test
            % fold
            net=patternnet(H,'trainscg');
            net.performFcn = 'crossentropy';
            net.divideFcn='divideind';
            ITRNComp=1:length(DIVISOR);
            ITRNComp(ITST)=[];
            % optimal training epochs
            net.trainParam.epochs=round(mean(val_epochs));
            net.divideParam.trainInd=ITRNComp;                          
            net.divideParam.valInd=[];                           
            net.divideParam.testInd=ITST;
            % Set the initial weight
            rng(S{i});
            net=configure(net,x,targets);
            [net,tr,y,e]=train(net,x,targets);
            perf_xentrtst(n,i)= tr.best_tperf;        
            perf_msetst(n,i)= mse(net,targets(:,ITST),y(:,ITST));
            perf_TSToutputs{n,i} = y(:,ITST);
            perf_TSTtargets{n,i} = targets(:,ITST);
            % accuracy
            [~,out]=max(y(:,ITST));
            [~,tar]=max(targets(:,ITST));
            cmat=confusionmat(out,tar);
            accuracy(n,i)=trace(cmat)/sum(cmat(:));
            for c=1:size(cmat,1)
                TP = cmat(c,c);               
                FP = sum(cmat(c,:))-TP;
                FN = sum(cmat(:,c))-TP;
                recall(n,i)=recall(n,i)+(TP/(TP+FN));
                precision(n,i)=precision(n,i)+(TP/(TP+FP));             
            end
            recall(n,i)=recall(n,i)/size(cmat,1);
            precision(n,i)=precision(n,i)/size(cmat,1);
            f1(n,i)=2*(precision(n,i)*recall(n,i))/(precision(n,i)+recall(n,i));  
        end
    end
    eval(['T',num2str(t),'Test_model.data.xentrval=perf_xentrval']);
    eval(['T',num2str(t),'Test_model.data.xentrtrn=perf_xentrtrn']);
    eval(['T',num2str(t),'Test_model.data.xentrtst=perf_xentrtst']);
    eval(['T',num2str(t),'Test_model.data.mseval=perf_mseval']);
    eval(['T',num2str(t),'Test_model.data.msetrn=perf_msetrn']);
    eval(['T',num2str(t),'Test_model.data.msetst=perf_msetst']);
    eval(['T',num2str(t),'Test_model.HiddenNeurons=Neurons']);
    eval(['T',num2str(t),'Test_model.data.epoch=perf_best_ep;']);
    eval(['T',num2str(t),'Test_model.data.outputs=perf_TSToutputs;']);
    eval(['T',num2str(t),'Test_model.data.targets=perf_TSTtargets;']);
    eval(['T',num2str(t),'Test_model.SET.Sbj=subgroup{t};']);
    eval(['T',num2str(t),'Test_model.SET.Ind=ITST;']);
    eval(['T',num2str(t),'Test_model.eval.Precision=precision;']);
    eval(['T',num2str(t),'Test_model.eval.Recall=recall;']);
    eval(['T',num2str(t),'Test_model.eval.Accuracy=accuracy;']);
    eval(['T',num2str(t),'Test_model.eval.F1=f1;']);
end