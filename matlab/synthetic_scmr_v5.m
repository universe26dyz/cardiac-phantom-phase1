function synthetic_scmr_v5(input_path, output_path, matcher_root)
% Synthetic observations with acquisition metadata supplied by the generator.
S=load(input_path,'Mag_synthetic','Acq_time','group_idx','weight_idx','source_sha256');
assert(size(S.Mag_synthetic,3)==10 && isequal(double(S.weight_idx(:))',(0:9)));
assert(isequal(double(S.group_idx(:))',0:size(S.Mag_synthetic,4)-1));
assert(size(S.Acq_time,1)==10 && size(S.Acq_time,2)==size(S.Mag_synthetic,4));
addpath(matcher_root); addpath(fullfile(matcher_root,'matlab','dictionary'));
matcher_path=which('function_T1T2_10HB_bssfp_qc'); simulator_path=which('sim_T1T2_10HB_bssfp');
assert(~isempty(matcher_path) && ~isempty(simulator_path));
[t1,t2,b1]=ndgrid([20:20:500,505:5:1500,1520:20:2500],[5:5:100,110:10:200],0.1:0.05:1.2);
Tlist=unique([t1(:),t2(:),b1(:),zeros(numel(t1),2)],'rows'); Tlist=Tlist(Tlist(:,1)>Tlist(:,2),:);
info=struct('RepetitionTime',2.61,'EchoTrainLength',87); n=size(S.Mag_synthetic,4);
T1=cell(n,1); T2=cell(n,1); B1=cell(n,1); QC=cell(n,1);
for k=1:n
 [T1{k},T2{k},B1{k},QC{k}]=function_T1T2_10HB_bssfp_qc(S.Mag_synthetic(:,:,:,k),info,S.Acq_time(:,k),[45,45,45],Tlist,[35,45,55],[10,100]);
end
save(output_path,'T1','T2','B1','QC','matcher_path','simulator_path','-v7');
end
