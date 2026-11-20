function run_mind_adapter(input_path, output_path, mind_source_dir)
addpath(mind_source_dir);
S=load(input_path);
assert(isfield(S,'noisy') && isfield(S,'clean'));
noisy=S.noisy; clean=S.clean;
assert(isequal(size(noisy),size(clean)) && ndims(noisy)==4);
[nr,nc,nh,ns]=size(noisy); assert(nh==10);
registered_noisy=zeros(size(noisy),'single');
registered_clean=zeros(size(clean),'single');
flow_u=zeros(nr,nc,nh,ns,'single'); flow_v=flow_u;
registration_qc=zeros(ns,nh,4,'single');
for s=1:ns
    fixed=single(noisy(:,:,1,s));
    registered_noisy(:,:,1,s)=fixed;
    registered_clean(:,:,1,s)=single(clean(:,:,1,s));
    for h=2:nh
        moving=single(noisy(:,:,h,s));
        [u,v,registered]=deformableReg2Dmind_asym_nodisplay(fixed,moving,0.3);
        flow_u(:,:,h,s)=u; flow_v(:,:,h,s)=v;
        registered_noisy(:,:,h,s)=registered;
        registered_clean(:,:,h,s)=warp_flow(single(clean(:,:,h,s)),u,v);
        registration_qc(s,h,1)=sqrt(mean((fixed(:)-moving(:)).^2));
        registration_qc(s,h,2)=sqrt(mean((fixed(:)-registered(:)).^2));
        registration_qc(s,h,3)=median(hypot(u(:),v(:)));
        registration_qc(s,h,4)=max(hypot(u(:),v(:)));
    end
end
save(output_path,'registered_noisy','registered_clean','flow_u','flow_v','registration_qc','-v7');
end
function B=warp_flow(A,u,v)
[m,n]=size(A); [x,y]=meshgrid(1:n,1:m);
xu=min(max(x+double(u),1),n); yv=min(max(y+double(v),1),m);
xu(isnan(xu))=0; yv(isnan(yv))=0;
B=interp2(A,xu,yv,'linear');
B(isnan(B))=A(isnan(B));
B=single(B);
end
