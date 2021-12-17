function pop = population(data, ids)
%     Extract a population for the original dataset.
% 
%     Parameters
%     ----------
%     data: Table
%       Dataset obtained with pandas.read_csv
%     ids:  Array of Int
%       List of ULiege ids for each group member (e.g. s167432 and s189134 -> [20167432,20189134])
% 
%     Returns
%     -------
%     Table containing your population

rng(sum(ids));
FIXED_COUNTRIES = {'USA', 'Belgium', 'China', 'Togo'};

pop = data(~ismember(data.Country,FIXED_COUNTRIES),:) ;
pop = datasample(pop,146,'Replace',false) ;
for i = 1:length(FIXED_COUNTRIES)
    pop = [pop;data(ismember(data.Country,FIXED_COUNTRIES{i}),:)];
end

end