function b = kron_mv(As, x)

D = length(As);
G = zeros(D,1);
for d = 1:D
    Ad = As{d};
    assert(size(Ad,1) == size(Ad,2));
    G(d) = size(Ad,1);
end
N = prod(G);

% %%%%%%%%%%%%%%%%%%%%%%%%%%
% X = reshape(x, G(1), round(N/G(1)));
% X = X';
% x = X(:);
% %%%%%%%%%%%%%%%%%%%%%%%%%%

b = x;
for dd = D:-1:1
    Ad = As{dd};
    X = reshape(b, G(dd), round(N/G(dd)));
    Y = Ad*X;
    b = Y';
    b = b(:);
end

% b = x;
% for dd = D:-1:1
%     X = reshape(b, round(N/G(dd)),G(dd));
%     Y = As{dd}*X;
%     b = Y';
%     b = b(:);
% end

    