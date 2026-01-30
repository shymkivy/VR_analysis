% trying reconstruct motion from sensor readings at 40deg from vertical to ground

% but, first trying to compute sensor readings from particular moutin

% phi is on horizontal plane, with zero facing us and goin counterclockwise into positive
% theta is on plane facing us with zero pointing up and positive counterclockwise
% s0 is at phi=0, theata=140
s0theta = 140;  % 140
s0phi = 0;
s1theta = 140;      %140;
s1phi = -90;        %270

theta = 0;
phi = 50;   % angle of vector defining rotation orientation with right hand rule
r = 1; % speed of rotation

ax = r * sind(theta) * cosd(phi);
ay = r * sind(theta) * sind(phi);
az = r * cosd(theta);

%  sensor direction is described with right hand rule
s00x = sind(90) * cosd(s0phi-90);
s00y = sind(90) * sind(s0phi-90);
s00z = cosd(90);

s01x = sind(s0theta + 90) * cosd(s0phi);
s01y = sind(s0theta + 90) * sind(s0phi);
s01z = cosd(s0theta + 90);

s10x = sind(90) * cosd(s1phi-90);
s10y = sind(90) * sind(s1phi-90);
s10z = cosd(90);

s11x = sind(s1theta + 90) * cosd(s1phi);
s11y = sind(s1theta + 90) * sind(s1phi);
s11z = cosd(s1theta + 90);

s00 = ax * s00x + ay * s00y + az * s00z;
s01 = ax * s01x + ay * s01y + az * s01z;
s10 = ax * s10x + ay * s10y + az * s10z;
s11 = ax * s11x + ay * s11y + az * s11z;

fprintf("input: theta = %.2f; phi = %.2f\n", theta, phi);
fprintf("s00 %.2f; s01 %.2f\n", s00, s01);
fprintf("s10 %.2f; s11 %.2f\n", s10, s11);


sens1 = [s00; s01; s10;s11];

mat1 = [s00x, s00y, s00z;
        s01x, s01y, s01z;
        s10x, s10y, s10z;
        s11x, s11y, s11z];

mat1_inv = pinv(mat1);

avals = mat1_inv*sens1;

avals = mat1\sens1;
%sens1'*mat1_inv'

r_out = avals'*avals;
phi_out = atan2d(avals(2), avals(1));
theta_out = acosd(avals(3)/r_out);

fprintf("recovered: theta = %.2f; phi = %.2f", theta_out, phi_out)

%s00 = r * sin((theta-50)/360*2*pi) * -sin(phi/360*2*pi);
%s01 = r * -cos(theta/360*2*pi)*sin(phi/360*2*pi);

%s00 = r * cos((theta-s0theta0)/360*2*pi) * -sin((phi-s0phi)/360*2*pi);
%s01 = r * sin((theta-s0theta0)/360*2*pi) * cos((phi-s0phi)/360*2*pi)

%s00 = r * sind(phi - s0phi);% * cosd(theta + s0theta);
%s01 = r * sind(theta - s0theta);

%s10 = r * sind(phi - s1phi);% * cosd(theta + s1theta);
%s11 = r * sind(theta - s1theta);

%s00 = r * cosd(theta - s0theta) * -sind(phi - s0phi);
%s01 = r * sind(theta - s0theta);

%s10 = r * cosd(theta - s1theta) * -sind(phi - s1phi);
%s11 = r * sind(theta - s1theta);

% s00 = r * sind(s0phi - phi) * cosd(s0theta - theta) * sind(theta);
% s01 = r * -sind(s0theta - theta);
% 
% s10 = r * sind(s1phi - phi) * cosd(s1theta - theta) * sind(theta);
% s11 = r * -sind(s1theta - theta);

