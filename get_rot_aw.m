function [a, w] = get_rot_aw(F)
% F = a .* R(w) where R is a rotation matrix with rotation radian w
assert(all(size(F)==[2,2]))
a = sqrt(F(1,1)^2+F(2,1)^2);
w = atan2(F(2,1),F(1,1));
end