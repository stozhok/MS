clear
X = double(imread('x3.bmp')); %вхідне зображення
Y = double(imread('y3.bmp')); %вихідне зображення
x1 = MoorePenrosePseudoInversion(X, 1);
x2 = GrevillePseudoInversion(X);
x1 = Y*x1;%оператор за допомогою методу Мура
x2 = Y*x2;%оператор за допомогою методу Гревіля
figure;
subplot(4,1,1);
imshow(uint8(X) );
title('INPUT');
subplot(4,1,2);
imshow(uint8(Y));
title('OUTPUT');
subplot(4,1,3);
imshow(uint8(x1*X) ); % перетворення вхідного зображення (оператор Мура)
title('MoorePenrosePseudoInversion');
subplot(4,1,4);
imshow(uint8(x2*X) );% перетворення вхідного зображення (оператор Гревіля)
title('GrevillePseudoInversion');

function n = norm(A)
    n = sum(sum(abs(A)));
end


function pInv = MoorePenrosePseudoInversion(A, delta)
    E = eye(size(A,1)); % одинична матриця
    invA0 = transpose(A); % транспонування
    invA = transpose(A) * inv(A * transpose(A) + delta * E); % застосування формули знаходження псевдооберненої матриці
    eps = norm(invA-invA0); % знаходження відхилення
    i = 1;
    while eps > 0.000001 % продовжуємо, поки не отримаємо мінімальне відхилення
    i = i + 1;
    invA0 = invA;
    delta = delta/2;
    invA = transpose(A) * inv(A * transpose(A) + delta * E);
    eps = norm(invA-invA0);
    end
pInv = invA;
end

function pInv = GrevillePseudoInversion(A)% метод Гревіля

    a = A(1,:)';% перший транспонований рядок
    res = 0;
    if (a' == 0)
        res = a;
    else
        res = a/(a' * a);     %res - кінцева псевдообернена матриця
    end
    sz = size(A,1); %рядки
    currentMatrix = a'; % матриця, утворена з послідовного додавання рядків

    for i = 2:sz  % проходимо всі рядки
        a = A(i,:)'; % і-ий транспонований рядок
        Z = eye(size(res,1)) - res * currentMatrix;
        R = res * res'; % параметри для формули Гревіля
        if(a' * Z * a > 0)
           res = [(res  - (Z * a * a' * res)/(a' * Z * a)),(Z * a)/(a' * Z * a)];
        else
            res = [(res  - (R * a * a' * res)/(1 + a' * R * a)),(R * a)/(1 + a' * R * a)];
        end % формула Гревіля
        currentMatrix = [currentMatrix;a']; % додавання рядка
    end

    pInv = res; %остаточний результат
end
