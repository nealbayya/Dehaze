## Copyright (C) 1999,2000  Kai Habel
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

## -*- texinfo -*-
## @deftypefn {Function File} @var{I}= mat2gray (@var{M},[min max])
## converts a matrix to a intensity image
## @end deftypefn

## Author:	Kai Habel <kai.habel@gmx.de>
## Date:	22/03/2000

function I = mat2gray (M, scale)

  if (nargin < 1|| nargin > 2)
    usage ("mat2gray(...) number of arguments must be 1 or 2");
  endif

  if (nargin == 1)
    Mmin = min (min (M));
    Mmax = max (max (M));
  else 
    if (is_vector (scale))
      Mmin = min (scale (1), scale (2));
      Mmax = max (scale (1), scale (2));
    endif
  endif

  I = (M < Mmin) .* 0;
  I = I + (M >= Mmin & M < Mmax) .* (1 / (Mmax - Mmin) * (M - Mmin));
  I = I + (M >= Mmax);

endfunction