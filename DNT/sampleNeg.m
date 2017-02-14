function negTmpl = sampleNeg(location, pf_param, param)
    c = 1;
    h = location(4);
    w = location(3);
%     sigma = [round(pf_param.p_sz*param.est(3)*param.p0), round(pf_param.p_sz*param.est(3)), .000, .000, .000, .000];
%     
%     back1 = round(sigma(1)/2);
%     center1 = param.param0(1,1);
%     left = center1 - back1;
%     right = center1 + back1;
%     
%     back2 = round(sigma(2)/2);
%     center2 = param.param0(1,2);
%     top = center2 - back2;
%     bottom = center2 + back2;
    
    for i = 1 : pf_param.num
        a = randn() * h / 6;
        b = randn() * w / 6;
        while abs(a) < h / 4 && abs(b) < w / 4
            a = randn() * h / 6;
            b = randn() * w / 6;
        end
        temp = location + [a,b,0,0];
        negTmpl(c, :) = floor(temp);
        c = c + 1;
    end
    
%     nono = negTmpl(:,1)<=right&negTmpl(:,1)>=center1;
%     negTmpl(nono,1) = right;
%     nono = negTmpl(:,1)>=left&negTmpl(:,1)<center1;
%     negTmpl(nono,1) = left;
%     
%     nono = negTmpl(:,2)<=bottom&negTmpl(:,2)>=center2;
%     negTmpl(nono,2) = bottom;
%     nono = negTmpl(:,2)>=top&negTmpl(:,2)<center2;
%     negTmpl(nono,2) = top;
end