tic
X=load('data.txt');
x1=X(1,:);
x2=X(2,:);
d=X(3,:);
n=length(x1);%样本个数  
p=6; %隐层个数  
w=rand(p,2);  
wk=rand(1,p+1);  
max_epoch=500000;%最大训练次数  
error_goal=0.00002;%均方误差  
q=0.2;%学习速率  
a(p+1)=-1;
  
%training  
%此训练网络采取2-6-1的形式，即两个输入，6个隐层，1个输出  
for epoch=1:max_epoch  
    e=0;  
    for i=1:n %样本个数  
        x=[x1(i);x2(i)];   
        neto=0;  
        for j=1:p   
            neti(j)=w(j,1)*x(1)+w(j,2)*x(2);  
            a(j)=1/(1+exp(-neti(j)));  
%隐层的激活函数采取s函数，f(x)=1/(1+exp(-x))  
            neto=neto+wk(j)*a(j);  
        end            
        neto=neto+wk(p+1)*(-1);
        y(i)=1/(1+exp(-neto));  %输出层的激活函数采取sigmoid函数  
        de=(1/2)*(d(i)-y(i))*(d(i)-y(i));  
        e=de+e;       
        dwk=q*(d(i)-y(i))*y(i)*(1-y(i))*a;   
        for k=1:p  
            dw(k,1:2)=q*(d(i)-y(i))*y(i)*(1-y(i))*wk(k)*a(k)*(1-a(k))*x;         
        end     
        wk=wk+dwk; %从隐层到输出层权值的更新  
        w=w+dw; %从输入层到隐层的权值的更新      
    end   
    error(epoch)=e;  
    m(epoch)=epoch;      
    if(e<error_goal)              
       break;  
    elseif(epoch==max_epoch)  
        disp('在目前的迭代次数内不能逼近所给函数，请加大迭代次数')          
    end   
end

%test data  
x1_te=[1.24,1.28,1.4];
x2_te=[1.8,1.84,2.04];
	
for i=1:3 %样本个数  
    x=[x1_te(i);x2_te(i);-1];    
    neto=0;  
    for j=1:p  
        neti(j)=w(j,1)*x(1)+w(j,2)*x(2);  
        a(j)=1/(1+exp(-neti(j)));  
        neto=neto+wk(j)*a(j);  
    end    
    neto=neto+wk(p+1)*(-1);  
    y1(i)=1/(1+exp(-neto));  %输出层的激活函数采取sigmoid函数
end   
y1(1:3)
toc
