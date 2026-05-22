function data=RF_process(data_input)
       for i=1:length(data_input)
            data(i,1)=str2double(data_input{i,1});
       end
end