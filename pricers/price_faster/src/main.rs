mod pricer;

use pricer::{JWSS7_STRUCT, _compute_from_jw7};

fn main() {
    let jw7_p = JWSS7_STRUCT((0.2, 0.3, 0.3, 0.4, 0.3, 0.4, 0.4));
    let _s_0 = 100.;
    let _r = 0.04;
    let _ttm = 1.;
    let _strikes_prices_cp = [
        (101, 2., "call"),
        (103, 4., "call"),
        (105, 7., "call"),
    ];

    print!("{}", _compute_from_jw7(0.1, &jw7_p) );

}
