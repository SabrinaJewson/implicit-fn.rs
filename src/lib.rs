//! A macro that adds support for implicit closures to Rust.
//!
//! This provides a concise alternative to regular closure syntax
//! for when each parameter is used at most once
//! and is not deeply nested.
//!
//! This feature [has been suggested before](https://github.com/rust-lang/rfcs/issues/2554),
//! but this macro mostly exists for fun.
//!
//! # Examples
//!
//! Using implicit closures to concisely sum a list:
//!
//! ```
//! #[implicit_fn]
//! fn main() {
//!     let n = [1, 2, 3].into_iter().fold(0, _ + _);
//!     assert_eq!(n, 6);
//! }
//! # use implicit_fn::implicit_fn;
//! ```
//!
//! Copying all the elements of an array:
//!
//! ```
//! #[implicit_fn]
//! fn main() {
//!     let array: [&u32; 3] = [&1, &2, &3];
//!     let array: [u32; 3] = array.map(*_);
//!     assert_eq!(array, [1, 2, 3]);
//! }
//! # use implicit_fn::implicit_fn;
//! ```
//!
//! Running a fallible function in an iterator:
//!
//! ```
//! #[implicit_fn]
//! fn main() -> Result<(), Box<dyn Error>> {
//!     let names = fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))?
//!         .map(_?.file_name().into_string().map_err(|_| "file not UTF-8")?)
//!         .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
//!     assert_eq!(names, ["lib.rs"]);
//!     Ok(())
//! }
//! # use implicit_fn::implicit_fn;
//! # use std::fs;
//! # use std::error::Error;
//! # use std::io;
//! ```
//!
//! Running a match on an array of options:
//!
//! ```
//! #[implicit_fn]
//! fn main() {
//!     let options = [Some(16), None, Some(2)];
//!     let numbers = options.map(match _ {
//!         Some(x) => x + 1,
//!         None => 0,
//!     });
//!     assert_eq!(numbers, [17, 0, 3]);
//! }
//! # use implicit_fn::implicit_fn;
//! ```
//!
//! Printing the elements of an iterator:
//!
//! ```
//! #[implicit_fn]
//! fn main() {
//!     [1, 2, 3].into_iter().for_each(println!("{}", _));
//! }
//! # use implicit_fn::implicit_fn;
//! ```
#![warn(clippy::pedantic)]
#![warn(redundant_lifetimes)]
#![warn(rust_2018_idioms)]
#![warn(single_use_lifetimes)]
#![warn(unit_bindings)]
#![warn(unused_crate_dependencies)]
#![warn(unused_lifetimes)]
#![warn(unused_qualifications)]
#![allow(clippy::items_after_test_module)]

/// Transform all the implicit closures inside the expressions of an item.
///
/// The size of the closure, i.e. where the `||` is put, is determined syntactically
/// according to the following rules:
/// + The closure is always large enough such that
///   identity functions (i.e. `|x| x`) are never generated,
///   except for `(_)` which is always an identity function.
/// + The closure encompasses as many “transparent” syntactic elements as it can,
///   where the transparent elements are
///   - unary and binary operators (e.g. prefix `*` and `!`, `/`, or `+=`);
///   - `.await`, `?`, field access, and
///     the left hand side of method calls and indexing expressions;
///   - the conditions of `while` and `if`,
///     the iterator in `for` and the scrutinee of a `match`, and
///   - both sides of a range expression (e.g. `_..=5`).
///
/// Notably, `f(_ + 5)` will always parse as `f(|x| x + 5)` and not `|x| f(x + 5)`,
/// and `(_ + 1) * 2` will parse as `(|x| x + 1) * 2`.
///
/// For examples, see [the crate docs](crate).
///
/// # Limitations
///
/// We only support `?` with [`Result`] and not other types implementing `Try`,
/// due to the lack of a stable `Try` trait.
///
/// Macro bodies are not transformed except for builtin macros.
#[proc_macro_attribute]
pub fn implicit_fn(
    attr: proc_macro::TokenStream,
    body: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    inner(attr.into(), body.into()).into()
}

fn inner(attr: TokenStream, body: TokenStream) -> TokenStream {
    let mut diagnostics = Diagnostics(None);

    if !attr.is_empty() {
        diagnostics.add(attr, "expected no tokens");
    }

    let mut item = match syn::parse2::<Item>(body) {
        Ok(input) => input,
        Err(e) => return diagnostics.with(e).into_compile_error(),
    };

    Transformer.visit_item_mut(&mut item);

    item.into_token_stream()
}

/// The top-level transformer.
struct Transformer;

impl VisitMut for Transformer {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        maybe_make_closure(e);
    }
}

/// Potentially make the given expression into a closure.
///
/// Also applies the transform to all subexpressions.
fn maybe_make_closure(e: &mut Expr) {
    if let Expr::Infer(_) = e {
        return;
    }
    maybe_make_closure_force(e);
}

fn maybe_make_closure_force(e: &mut Expr) {
    let mut helper = ReplaceHoles {
        is_async: None,
        is_try: None,
        holes: Vec::new(),
    };
    replace_holes(&mut helper, e);
    if helper.holes.is_empty() {
        return;
    }

    let or1_span = helper.holes.first().unwrap().span();
    let or2_span = helper.holes.last().unwrap().span();
    let mut inputs = helper
        .holes
        .into_iter()
        .map(|id| {
            let span = id.span();
            punctuated::Pair::Punctuated(ident_pat(id), Token![,](span))
        })
        .collect::<Punctuated<Pat, Token![,]>>();
    inputs.pop_punct();

    let old_expr = replace(e, Expr::PLACEHOLDER);
    *e = Expr::Closure(ExprClosure {
        attrs: Vec::new(),
        lifetimes: None,
        constness: None,
        movability: None,
        asyncness: helper.is_async.map(Token![async]),
        capture: None,
        or1_token: Token![|](or1_span),
        inputs,
        or2_token: Token![|](or2_span),
        output: ReturnType::Default,
        body: match helper.is_try {
            Some(try_span) => Box::new(Expr::Verbatim(
                quote_spanned!(try_span=> ::core::result::Result::Ok(#old_expr)),
            )),
            None => Box::new(old_expr),
        },
    });
}

/// Traverses an expression’s subexpressions,
/// ignoring subexpressions transparent to the macro,
/// replacing holes with variables.
struct ReplaceHoles {
    is_async: Option<Span>,
    is_try: Option<Span>,
    holes: Vec<Ident>,
}

fn replace_holes(st: &mut ReplaceHoles, e: &mut Expr) {
    match e {
        Expr::Await(e) => {
            st.is_async.get_or_insert(e.await_token.span);
            replace_holes(st, &mut e.base);
        }
        Expr::Binary(e) => {
            replace_holes(st, &mut e.left);
            replace_holes(st, &mut e.right);
        }
        Expr::Call(e) => {
            replace_holes(st, &mut e.func);
        }
        Expr::Field(e) => replace_holes(st, &mut e.base),
        Expr::ForLoop(e) => replace_holes(st, &mut e.expr),
        Expr::If(e) => {
            replace_holes(st, &mut e.cond);
            if let Some((_, else_branch)) = &mut e.else_branch {
                replace_holes(st, else_branch);
            }
        }
        Expr::Index(e) => replace_holes(st, &mut e.expr),
        Expr::Let(e) => replace_holes(st, &mut e.expr),
        Expr::Match(e) => replace_holes(st, &mut e.expr),
        Expr::MethodCall(e) => replace_holes(st, &mut e.receiver),
        Expr::Range(e) => {
            if let Some(start) = &mut e.start {
                replace_holes(st, start);
            }
            if let Some(end) = &mut e.end {
                replace_holes(st, end);
            }
        }
        Expr::Try(e) => {
            st.is_try.get_or_insert(e.question_token.span);
            replace_holes(st, &mut e.expr);
        }
        Expr::Unary(e) => replace_holes(st, &mut e.expr),
        Expr::While(e) => replace_holes(st, &mut e.cond),

        Expr::Infer(infer) => *e = st.add(infer.underscore_token),

        // As an exception to the “no identity closures” rule,
        // we allow (_) itself to be an identity closure.
        Expr::Paren(e) => return maybe_make_closure_force(&mut e.expr),

        Expr::Assign(e) => {
            struct Helper<'a>(&'a mut ReplaceHoles);
            impl VisitMut for Helper<'_> {
                fn visit_expr_mut(&mut self, e: &mut Expr) {
                    replace_holes(self.0, e);
                }
            }
            IgnoreAssigneeExpr(Helper(st)).visit_expr_mut(&mut e.left);
            st.visit_expr_mut(&mut e.right);
            return;
        }

        _ => {}
    }
    // Call the freestanding function instead of the method
    // to look for `_`s in top-level expressions.
    visit_expr_mut(st, e);
}

// `VisitMut` implementation for all the top-level expressions,
// to avoid producing identity closures.
impl VisitMut for ReplaceHoles {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        if let Expr::Infer(infer) = e {
            *e = self.add(infer.underscore_token);
        } else {
            maybe_make_closure_force(e);
        }
    }
    // Override to treat `let _: [T; _] = …;` properly.
    fn visit_type_array_mut(&mut self, i: &mut TypeArray) {
        maybe_make_closure(&mut i.len);
    }
    // Override to treat `f::<{ _ }>` properly.
    fn visit_generic_argument_mut(&mut self, a: &mut GenericArgument) {
        if let GenericArgument::Const(Expr::Block(block)) = a {
            self.visit_expr_block_mut(block);
            return;
        }
        visit_generic_argument_mut(self, a);
    }
    fn visit_macro_mut(&mut self, m: &mut Macro) {
        visit_macro(self, m);
    }
}

impl ReplaceHoles {
    fn add(&mut self, underscore_token: Token![_]) -> Expr {
        let n = self.holes.len();
        let ident = Ident::new(&format!("p{n}"), underscore_token.span);
        self.holes.push(ident.clone());
        Expr::Verbatim(ident.into_token_stream())
    }
}

fn ident_pat(ident: Ident) -> Pat {
    Pat::Ident(PatIdent {
        attrs: Vec::new(),
        by_ref: None,
        mutability: None,
        ident,
        subpat: None,
    })
}

struct IgnoreAssigneeExpr<V>(V);

impl<V: VisitMut> VisitMut for IgnoreAssigneeExpr<V> {
    // For the definition of assignee expressions:
    // https://doc.rust-lang.org/reference/expressions.html#place-expressions-and-value-expressions
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        match e {
            Expr::Infer(_) => {}
            Expr::Tuple(e) => {
                for elem in &mut e.elems {
                    self.visit_expr_mut(elem);
                }
            }
            Expr::Array(e) => {
                for elem in &mut e.elems {
                    self.visit_expr_mut(elem);
                }
            }
            Expr::Call(e) => {
                self.0.visit_expr_mut(&mut e.func);
                for arg in &mut e.args {
                    self.visit_expr_mut(arg);
                }
            }
            Expr::Struct(e) => {
                for field in &mut e.fields {
                    self.visit_field_value_mut(field);
                }
                if let Some(rest) = &mut e.rest {
                    self.visit_expr_mut(rest);
                }
            }
            _ => self.0.visit_expr_mut(e),
        }
    }
}

#[derive(Default)]
struct Diagnostics(Option<syn::Error>);

impl Diagnostics {
    fn add_error(&mut self, new_e: syn::Error) {
        *self = Self(Some(take(self).with(new_e)));
    }
    fn with(self, mut new_e: syn::Error) -> syn::Error {
        if let Some(mut e) = self.0 {
            e.combine(new_e);
            new_e = e;
        }
        new_e
    }
    fn add(&mut self, tokens: impl ToTokens, message: impl Display) {
        self.add_error(syn::Error::new_spanned(tokens, message));
    }
}

fn visit_macro(v: &mut impl VisitMut, m: &mut Macro) {
    let _ = visit_macro_inner(v, m);
}

fn visit_macro_inner(v: &mut impl VisitMut, m: &mut Macro) -> syn::Result<()> {
    let Some(ident) = m.path.get_ident() else {
        return Ok(());
    };
    let ident = ident.to_string();

    if [
        "assert",
        "assert_eq",
        "assert_ne",
        "dbg",
        "debug_assert",
        "debug_assert_eq",
        "debug_assert_ne",
        "eprint",
        "eprintln",
        "format",
        "format_args",
        "panic",
        "print",
        "println",
        "todo",
        "unimplemented",
        "unreachable",
        "vec",
        "write",
        "writeln",
    ]
    .contains(&&*ident)
    {
        let mut args = m.parse_body_with(|input: ParseStream<'_>| {
            <Punctuated<Expr, Token![,]>>::parse_terminated(input)
        })?;
        for arg in &mut args {
            v.visit_expr_mut(arg);
        }
        m.tokens = args.into_token_stream();
    } else if ident == "matches" {
        let (mut scrutinee, comma, pattern, mut guard, comma2) =
            m.parse_body_with(|input: ParseStream<'_>| {
                let scrutinee: Expr = input.parse()?;
                let comma = input.parse::<Token![,]>()?;
                let pattern = Pat::parse_multi(input)?;
                let guard = match input.parse::<Option<Token![if]>>()? {
                    Some(r#if) => Some((r#if, input.parse::<Expr>()?)),
                    None => None,
                };
                let comma2 = input.parse::<Option<Token![,]>>()?;
                Ok((scrutinee, comma, pattern, guard, comma2))
            })?;
        v.visit_expr_mut(&mut scrutinee);
        if let Some((_, guard)) = &mut guard {
            v.visit_expr_mut(guard);
        }

        m.tokens = TokenStream::new();
        scrutinee.to_tokens(&mut m.tokens);
        comma.to_tokens(&mut m.tokens);
        pattern.to_tokens(&mut m.tokens);
        if let Some((r#if, guard)) = guard {
            r#if.to_tokens(&mut m.tokens);
            guard.to_tokens(&mut m.tokens);
        }
        comma2.to_tokens(&mut m.tokens);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn top_level_underscores() {
        assert_output!(_, _);
        assert_output!(
            {
                _;
            },
            |p0| {
                p0;
            }
        );
    }

    #[test]
    fn identity() {
        assert_output!((_), (|p0| p0));
        assert_output!({ { _ } }, { |p0| { p0 } });
    }

    #[test]
    fn avoid_identity() {
        assert_output!(f(_), |p0| f(p0));
        assert_output!((f(_)), (|p0| f(p0)));
        assert_output!({ _ }, |p0| { p0 });
        assert_output!(
            if x {
                _
            },
            |p0| if x {
                p0
            },
        );
        assert_output!(|_| _, |p0| |_| p0);
        assert_output!(
            {
                x;
                _
            },
            |p0| {
                x;
                p0
            },
        );
        assert_output!(
            {
                _;
                _
            },
            |p0, p1| {
                p0;
                p1
            },
        );
        assert_output!(
            if true {
                _;
                _
            },
            |p0, p1| if true {
                p0;
                p1
            },
        );
    }

    #[test]
    fn assignee_expression_underscores() {
        assert_output!(_ = x, _ = x);
        assert_output!((_, _) = x, (_, _) = x);
        assert_output!([_] = x, [_] = x);
        assert_output!(f(_) = x, f(_) = x);
        assert_output!(S { x: _ } = x, S { x: _ } = x);
    }

    #[test]
    fn assignment() {
        assert_output!(*f(_) = x, |p0| *f(p0) = x);
        assert_output!(*f(g(_)) = x, *f(|p0| g(p0)) = x);
        assert_output!(x = _, |p0| x = p0);
        assert_output!(x = f(_), x = |p0| f(p0));
    }

    #[test]
    fn infer_underscores() {
        assert_output!(
            {
                let x: [u8; _];
            },
            {
                let x: [u8; _];
            }
        );
        assert_output!(
            {
                let x: [u8; f(_)];
            },
            {
                let x: [u8; |p0| f(p0)];
            }
        );
        assert_output!(f::<_>(), f::<_>());
        assert_output!(f::<{ _ }>(), |p0| f::<{ p0 }>());
        assert_output!(f::<{ f(_) }>(), f::<{ |p0| f(p0) }>());
    }

    #[test]
    fn transparent() {
        assert_output!(f(_).await, async |p0| f(p0).await);
        assert_output!(f(_) + 1, |p0| f(p0) + 1);
        assert_output!(x += f(_), |p0| x += f(p0));
        assert_output!(f(_).f, |p0| f(p0).f);
        assert_output!(for _ in f(_) {}, |p0| for _ in f(p0) {});
        assert_output!(if f(_) {}, |p0| if f(p0) {});
        assert_output!(
            if x {
            } else if f(_) {
            },
            |p0| if x {
            } else if f(p0) {
            }
        );
        assert_output!(f(_)[0], |p0| f(p0)[0]);
        assert_output!(if let P = f(_) {}, |p0| if let P = f(p0) {});
        assert_output!(match f(_) {}, |p0| match f(p0) {});
        assert_output!(f(_).g(), |p0| f(p0).g());
        assert_output!(_..x, |p0| p0..x);
        // This doesn’t even parse for some reason:
        // assert_output!(x..=_, |p0| x..p0);
        assert_output!(f(_)?, |p0| ::core::result::Result::Ok(f(p0)?));
        assert_output!(!f(_), |p0| !f(p0));
        assert_output!(while f(_) {}, |p0| while f(p0) {});
    }

    #[test]
    fn macros() {
        assert_output!(arbitrary!(_), arbitrary!(_));
        assert_output!(assert!(_), |p0| assert!(p0));
        assert_output!(assert_ne!(_, _, _), |p0, p1, p2| assert_ne!(p0, p1, p2));
        assert_output!(vec![_], |p0| vec![p0]);
        assert_output!(matches!(_, _), |p0| matches!(p0, _));
        assert_output!(matches!(_, _ if _), |p0, p1| matches!(p0, _ if p1));
    }

    macro_rules! assert_output {
        ($in:expr, $out:expr $(,)?) => {
            assert_output_inner(quote!($in), quote!($out))
        };
    }
    use assert_output;

    #[track_caller]
    fn assert_output_inner(r#in: TokenStream, out: TokenStream) {
        let mut r#in = syn::parse2::<Expr>(r#in).unwrap();
        let out = syn::parse2::<Expr>(out).unwrap();
        maybe_make_closure(&mut r#in);
        assert_eq!(
            r#in.into_token_stream().to_string(),
            out.into_token_stream().to_string()
        );
    }

    use super::maybe_make_closure;
    use proc_macro2::TokenStream;
    use quote::ToTokens as _;
    use quote::quote;
    use syn::Expr;
}

use proc_macro2::Ident;
use proc_macro2::Span;
use proc_macro2::TokenStream;
use quote::ToTokens;
use quote::quote_spanned;
use std::fmt::Display;
use std::mem::replace;
use std::mem::take;
use syn::Expr;
use syn::ExprClosure;
use syn::GenericArgument;
use syn::Item;
use syn::Macro;
use syn::Pat;
use syn::PatIdent;
use syn::ReturnType;
use syn::Token;
use syn::TypeArray;
use syn::parse::ParseStream;
use syn::punctuated;
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::visit_mut::visit_expr_mut;
use syn::visit_mut::visit_generic_argument_mut;
