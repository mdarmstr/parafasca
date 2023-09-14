function parafascao = parafasca(parglmo,params,Rt)
%% PARAFAC - ANOVA Simultaneous Component Analysis (PARAFASCA)
%
% INPUTS
% parglmo - output of parglm.m in MEDA toolbox
% Rt      - Tensor rank used for PARAFAC modelling
%
% OUTPUTS
% parafascao - parafactors{mode}(entry,parafac_component,replicate)
%                  lbls{factors}
%                  iter := iterations
%                  err_glm := residuals of the PARAFAC model
%                  corr := core consistency diagnostic (0-100)
%
% Software preparation:  Install MEDA-Toolbox following readme file;
%                       Install the N-way toolbox following readme file;
%
%
% EXAMPLE OF USE (copy and paste the code in the command line)
%   Random data, two factors with 4 and 3 levels, and 4 replicates, with 
%   significant interaction (adapted from parglm.m)
% 
% reps = 4;
% vars = 400;
% levels = {[1,2,3,4],[1,2,3]};
% 
% F = create_design(levels,reps);
% 
% X = zeros(size(F,1),vars);
% for i = 1:length(levels{1}),
%     for j = 1:length(levels{2}),
%         X(find(F(:,1) == levels{1}(i) & F(:,2) == levels{2}(j)),:) = simuleMV(reps,vars,8) + repmat(randn(1,vars),reps,1);
%     end
% end
% 
% [tbl,parglmo] = parglm(X, F, {[1 2]});
%
% parafascao = parafasca(parglmo,2);
%
% coded by: Michael Sorochan Armstrong (mdarmstr@ugr.es)
%           Jose Camacho Paez (josecamacho@ugr.es)
%
% last modification: /Feb/2023
%
% Copyright (C) 2023  University of Granada, Granada
% Copyright (C) 2023  Jose Camacho Paez, Michael Sorochan Armstrong
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% MEDA dependencies: parglmo, create_design
% n-way_toolbox dependencies: parafac, n_fold

% if parglmo.n_factors == 0 && parglmo.n_interactions == 0
%     error('parglm model does not contain factors or interactions')
% end

Xa = zeros(size(parglmo.data));
Xm = zeros(size(parglmo.data));
Xe = zeros(size(parglmo.data));

[~,ncols] = cellfun(@size,params);
main_effects = cell2mat(params(ncols==1));
intr_effects = params(ncols>1);

for ii = 1:parglmo.n_factors
    if sum(ismember(ii,main_effects)) == 1
        Xa = Xa + parglmo.factors{ii}.matrix;
        main_effects(ii) = 0;
    else
        Xe = Xe + parglmo.factors{ii}.matrix; 
        main_effects(ii) = 0;
    end
end

for ii = 1:size(intr_effects,2)
    permuts = perms(intr_effects{ii});
    if sum(ismember(permuts,intr_effects{ii},'Rows')) == 1
        Xm = Xm + parglmo.interactions{ii}.matrix;
        intr_effects{ii} = [];
    else
        Xe = Xe + parglmo.interactions{ii}.matrix;
        intr_effects{ii} = [];
    end
end

nmode = size(parglmo.design,2);
Xs = Xm + Xa;
Xe = Xe + parglmo.residuals;

[Xc, Xce, Fc] = incomplete_factorial(Xs,Xe,parglmo.design);

[Xf_m,Xfe_m,lbls] = fold_factorial(Xc,Xce,Fc);

incl = 1:nmode + 1; %not accounting for replicate
idx = repmat({':'}, 1, nmode + 2);
rps = num_reps(Fc);
szt = size(Xf_m);

idx_excl_reps = idx;
idx_excl_reps{end} = 1;

Xf_n = Xf_m(idx_excl_reps{:});

[Fm,iter,err,corr] = parafac(Xf_n,Rt,[0,0,0,2,0,0]);

parafascao = parafasca_fitres(Fm, Xf_n, Xfe_m, nmode, incl, rps, szt, idx, Rt);

parafascao.lbls = lbls;

parafascao.iter = iter;

parafascao.err_glm = err;

parafascao.corr = corr;

end

function parafascao = parafasca_fitres(Fm,Xf_n,Xfe_m,nmode,incl,rps,szt,idx,Rt)
%% Fits the PARAFAC solution to the GLM data to the residual error tensor.

parafascao = struct();

for ii = 1:nmode+1

    incl_t = incl;
    incl_t(ii) = [];

    for jj = 1:nmode-1
        if jj == 1
            Z = krb(Fm{incl_t(jj+1)},Fm{incl_t(jj)});
        else
            Z = krb(Fm{incl_t(jj+1)},Z);
        end
    end

    parafascao.parafactors{ii} = zeros(szt(ii),Rt,rps);

    for jj = 1:rps

        idx{end} = jj;

        Xi = nshape(Xf_n + Xfe_m(idx{:}),ii);

        parafascao.parafactors{ii}(:,:,jj) = Xi * Z * pinv(Z'*Z); %Not optimised here

    end
end

end

function [Xf,Xfe,lbls] = fold_factorial(Xc,Xce,Fc)
%% Folds according to factorial design.
% Only use with full factorial design matrices.
% The last index will always be the number of replicates.
% TODO: how to check efficiently for a full factorial matrix

nmode = size(Fc,2);

levels = cell(nmode,1);
factor_levels = cell(nmode,1); % Number of factors = number of columns

for ii = 1:nmode
    factor_levels{ii} = unique(Fc(:,ii))';
    levels{ii} = 1:length(factor_levels{ii});
end

indices = create_design(levels,1);

indices_f = indices;

for ii = 1:nmode
    for jj = 1:length(factor_levels{ii})
        indices_f(indices(:,ii) == levels{ii}(jj),ii) = factor_levels{ii}(jj);
    end
end

%indices = recursive_index(sz);    

%number of replicates
rp = num_reps(Fc);

vr = size(Xc,2); %number of variables
sz = cellfun(@max,levels)'; %maximum levels 
szt = [sz,vr,rp]; %dimensions of tensor

Xf = zeros(szt);
Xfe = zeros(szt);

p = cumprod([1 szt(1:end-1)]);

for ii = 1:length(indices)
    logc_ar = find(all(Fc == indices_f(ii,:),2));
    for jj = 1:rp
        indx_rp = logc_ar(jj);
        for kk = 1:vr
            subind = [indices(ii,:),kk,jj];
            ind = (subind - 1) * p(:) + 1;
            Xf(ind) = Xc(indx_rp,kk);
            Xfe(ind) = Xce(indx_rp,kk);
        end
    end
end

lbls = cell(length(sz),1);

for ii = 1:length(sz)
    lbls{ii} = unique(Fc(:,ii));
end

end

function rp = num_reps(Fc)
[unique_rows,~,row_idx] = unique(Fc,'rows');
counts = histcounts(row_idx, 1:numel(unique_rows)+1);
rp = max(counts);
end

% function indices = recursive_index(dimensions,current_index)
% 
% if nargin < 2
%     current_index = [];
% end
% 
% if isempty(dimensions)
%     indices = {current_index};
%     return;
% end
% 
% indices = {};
% for ii = 1:dimensions(1)
%     new_index = [current_index, ii];
%     sub_indices = recursive_index(dimensions(2:end),new_index);
%     indices = [indices,sub_indices]; %#ok 
% end
% 
% end

function [Xc,Xce,Fc] = incomplete_factorial(X,Xe,F)
%%check for incomplete factorial design, return sparse factorial matrix and
%%complete design matrix

% calculate factors, levels of the experimental design
levels = cell(size(F,2),1);

for ii = 1:size(F,2)
    levels{ii} = unique(F(:,ii));
end

% calculate the maximum number of replicates

[~, ~, ID] = unique(F,"rows");
counts = histcounts(ID,1:max(ID)+1);

reps = max(counts);

Fc = create_design(levels,reps);

Xc = zeros(size(Fc,1),size(X,2));
Xce = zeros(size(Fc,1),size(X,2));

[a,~,a2] = unique([Fc;F],'rows');

identifier_new = a2(1:size(Fc,1));
identifier_old = a2((size(Fc,1)+1):end);

for ii = 1:size(a,1)
    if ismember(ii,identifier_old)
        strt_nc = find(identifier_new == ii,1);
        count_u = sum(identifier_old == ii) - 1;
        Xc(strt_nc:strt_nc+count_u,:) = X(identifier_old == ii, :);
        Xce(strt_nc:strt_nc+count_u,:) = Xe(identifier_old == ii, :);
    end
end

end
