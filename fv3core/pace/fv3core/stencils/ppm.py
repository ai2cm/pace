from gt4py.cartesian import gtscript

from pace.dsl.typing import FloatField


# volume-conserving cubic with 2nd drv=0 at end point:
# non-monotonic
c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0

# PPM volume mean form
p1 = 7.0 / 12.0
p2 = -1.0 / 12.0

s11 = 11.0 / 14.0
s14 = 4.0 / 7.0
s15 = 3.0 / 14.0


@gtscript.function
def pert_ppm_standard_constraint_fcn(a0: FloatField, al: FloatField, ar: FloatField):
    if al * ar < 0.0:
        da1 = al - ar
        da2 = da1 ** 2
        a6da = 3.0 * (al + ar) * da1
        if a6da < -da2:
            ar = -2.0 * al
        elif a6da > da2:
            al = -2.0 * ar
    else:
        # effect of dm=0 included here
        al = 0.0
        ar = 0.0
    return al, ar


@gtscript.function
def pert_ppm_positive_definite_constraint_fcn(
    a0: FloatField, al: FloatField, ar: FloatField
):
    if a0 <= 0.0:
        al = 0.0
        ar = 0.0
    else:
        a4 = -3.0 * (ar + al)
        da1 = ar - al
        if abs(da1) < -a4:
            fmin = a0 + 0.25 / a4 * da1 ** 2 + a4 * (1.0 / 12.0)
            if fmin < 0.0:
                if ar > 0.0 and al > 0.0:
                    ar = 0.0
                    al = 0.0
                elif da1 > 0.0:
                    ar = -2.0 * al
            else:
                al = -2.0 * ar

    return al, ar
