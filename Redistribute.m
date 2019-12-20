function [groups]=Redistribute(D,groups,R)

occurencesG=sortrows([cell2mat(cellfun(@(x)sum(ismember(D,x)),groups,'un',0)),(1:1:length(groups))'],1,'ascend');
occurencesR=sortrows([cell2mat(cellfun(@(x)sum(ismember(D,x)),R,'un',0)),(1:1:length(R))'],1,'descend');
for k=1:length(R)
    groups{occurencesG(k,2)}=[groups{occurencesG(k,2)};R{occurencesR(k,2)}];
end

