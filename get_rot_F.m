function [F] = get_rot_F(a,w)
% F = a .* R(w) where R is a rotation matrix with rotation radian w
F = zeros(2,2);
F(1,1) = a * cos(w);
F(1,2) = a * -sin(w);
F(2,1) = a * sin(w);
F(2,2) = a * cos(w);
end
