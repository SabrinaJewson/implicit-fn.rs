#[test]
#[implicit_fn]
fn statement_macro() {
    [true, true].map({
        assert!(_);
    });
}

use implicit_fn::implicit_fn;
