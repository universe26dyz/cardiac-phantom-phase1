function run_synthetic_groups(directory, matcher_root)
% Run the frozen core once per synthetic slice; do not replace existing outputs.
names={'sax','2ch','4ch'}; counts=[12,15,10];
for k=1:numel(names)
 for j=0:counts(k)-1
  stem=fullfile(directory,'groups',sprintf('%s_%02d',names{k},j));
  input_path=[stem '_input.mat']; output_path=[stem '_output.mat'];
  if ~isfile(output_path)
   synthetic_scmr_v5(input_path,output_path,matcher_root);
  end
  fprintf('%s %d complete\n',names{k},j);
 end
end
end
