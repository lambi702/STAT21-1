function out = scientific_delta(pop)

%     Parameters
%     ----------
%     pop: Table
%         Table containing a column 'PIB/habitant' and 'CO2/habitant'
% 
%     Returns
%     -------
%     Delta value computed by scientists

    median_gdp = median( pop.PIB_habitant ) ;
    
    pop_rich = pop(pop.PIB_habitant >= median_gdp,:);
    pop_poor = pop(pop.PIB_habitant <  median_gdp,:);

    out = mean(pop_rich.CO2_habitant) - mean(pop_poor.CO2_habitant);
end