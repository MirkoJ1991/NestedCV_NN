function [val_xentrval,val_xentrtrn,val_mseval,val_msetrn,val_epochs]=ValidationLoops(S,folds,x,targets,H,ITRN,IVAL,ITST)
% Initialize data
val_xentrval = zeros(1,folds-1);                     
val_xentrtrn = zeros(1,folds-1);                                       
val_mseval = zeros(1,folds-1);                      
val_msetrn = zeros(1,folds-1);                      
val_epochs = zeros(1,folds-1);
% For each validation fold
for v=1:folds-1
    % Define the net
    net=patternnet(H,'trainscg');
    net.performFcn = 'crossentropy';
    net.divideFcn='divideind';
    net.divideParam.trainInd=ITRN{v};                       
    net.divideParam.valInd=IVAL{v};                            
    net.divideParam.testInd=ITST;
    % Set the initial weight random state
    rng(S);
    net=configure(net,x,targets);
    [net,tr,y,e]=train(net,x,targets);

    val_xentrval(v) = tr.best_vperf;
    val_xentrtrn(v) = tr.best_perf;
    val_epochs(v) = tr.best_epoch;
    val_mseval(v) = mse(net,targets(:,IVAL{v}),y(:,IVAL{v}));              
    val_msetrn(v) = mse(net,targets(:,ITRN{v}),y(:,ITRN{v}));
end