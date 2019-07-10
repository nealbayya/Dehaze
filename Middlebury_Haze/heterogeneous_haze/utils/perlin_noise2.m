function s = perlin_noise2(m)
  s = zeros([m,m]);     % Prepare output image (size: m x m)
  w = m;
  i = 0;
  while w > 3
    i = i + 1;
%     d = interp2(randn([m,m]), i-1, 'spline');
    d = double(gaussf(randn([m,m]),i));
    s = s + i * d(1:m, 1:m);
    w = w - ceil(w/2 - 1);
  end
  s = dip_image(s - min(min(s(:,:)))) ./ (max(max(s(:,:))) - min(min(s(:,:))));
end