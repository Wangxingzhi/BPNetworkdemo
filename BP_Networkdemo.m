tic
X=load('data.txt');
x1=X(1,:);
x2=X(2,:);
d=X(3,:);
n=length(x1);%��������  
p=6; %�������  
w=rand(p,2);  
wk=rand(1,p+1);  
max_epoch=500000;%���ѵ������  
error_goal=0.00002;%�������  
q=0.2;%ѧϰ����  
a(p+1)=-1;
  
%training  
%��ѵ�������ȡ2-6-1����ʽ�����������룬6�����㣬1�����  
for epoch=1:max_epoch  
    e=0;  
    for i=1:n %��������  
        x=[x1(i);x2(i)];   
        neto=0;  
        for j=1:p   
            neti(j)=w(j,1)*x(1)+w(j,2)*x(2);  
            a(j)=1/(1+exp(-neti(j)));  
%����ļ������ȡs������f(x)=1/(1+exp(-x))  
            neto=neto+wk(j)*a(j);  
        end            
        neto=neto+wk(p+1)*(-1);
        y(i)=1/(1+exp(-neto));  %�����ļ������ȡsigmoid����  
        de=(1/2)*(d(i)-y(i))*(d(i)-y(i));  
        e=de+e;       
        dwk=q*(d(i)-y(i))*y(i)*(1-y(i))*a;   
        for k=1:p  
            dw(k,1:2)=q*(d(i)-y(i))*y(i)*(1-y(i))*wk(k)*a(k)*(1-a(k))*x;         
        end     
        wk=wk+dwk; %�����㵽�����Ȩֵ�ĸ���  
        w=w+dw; %������㵽�����Ȩֵ�ĸ���      
    end   
    error(epoch)=e;  
    m(epoch)=epoch;      
    if(e<error_goal)              
       break;  
    elseif(epoch==max_epoch)  
        disp('��Ŀǰ�ĵ��������ڲ��ܱƽ�������������Ӵ��������')          
    end   
end

%test data  
x1_te=[1.24,1.28,1.4];
x2_te=[1.8,1.84,2.04];
	
for i=1:3 %��������  
    x=[x1_te(i);x2_te(i);-1];    
    neto=0;  
    for j=1:p  
        neti(j)=w(j,1)*x(1)+w(j,2)*x(2);  
        a(j)=1/(1+exp(-neti(j)));  
        neto=neto+wk(j)*a(j);  
    end    
    neto=neto+wk(p+1)*(-1);  
    y1(i)=1/(1+exp(-neto));  %�����ļ������ȡsigmoid����
end   
y1(1:3)
toc
