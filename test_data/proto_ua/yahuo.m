clear;clc;

load('C:\Users\user\Desktop\程序\matlab程序\预处理ua\p0\p0_init.mat')
load('C:\Users\user\Desktop\程序\matlab程序\预处理ua\p0\p0_log.mat')
load('C:\Users\user\Desktop\程序\matlab程序\预处理ua\p0\p0_n.mat')
load('C:\Users\user\Desktop\程序\matlab程序\预处理ua\p0\p0_inf.mat')

load('C:\Users\user\Desktop\程序\matlab程序\预处理ua\fai\真实fai\fai_init.mat')
load('C:\Users\user\Desktop\程序\matlab程序\预处理ua\fai\真实fai\fai_log.mat')
load('C:\Users\user\Desktop\程序\matlab程序\预处理ua\fai\真实fai\fai_nor.mat')
load('C:\Users\user\Desktop\程序\matlab程序\预处理ua\fai\真实fai\fai_inf.mat')

load('C:\Users\user\Desktop\程序\matlab程序\预处理ua\ua\ua_true.mat')


p0_1=p0_n*14.3986;
p0_2=exp(p0_1);

ua_1=p0_2./fai_init;
ua_2=ua_1;
ua_2(find(ua_1==inf))=0;
% kkk=log10(fai_init);


