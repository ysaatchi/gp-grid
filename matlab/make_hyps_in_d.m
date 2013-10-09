function hyps_in_d = make_hyps_in_d(Qs,cov)
P = length(Qs);
hyps_in_d = cell(P,1);
Qt = sum(Qs);
windex = 0;
muindex = Qt;
sindex = 2*Qt;

switch(cov{1})
case 'covSM1D'
for p=1:P
Q_in_d = Qs(p);
hyps_in_d{p} = [windex+1:windex+Q_in_d,muindex+1:muindex+Q_in_d,sindex+1:sindex+Q_in_d]';
windex = windex+Q_in_d;
muindex = muindex+Q_in_d;
sindex = sindex+Q_in_d;
end
case {'covSEard','covMaterniso'}
for p=1:P
hyps_in_d{p} = [p,P+1]';
end
case 'covRQard'
for p=1:P
hyps_in_d{p} = [p,P+1,P+2]';
end
end