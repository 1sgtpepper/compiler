use std::{env, fs, path::PathBuf};

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote};
use syn::{
    Error, FnArg, Item, ItemFn, LitStr, Pat, Token, TypePath,
    parse::{Parse, ParseStream},
    spanned::Spanned,
    visit_mut::VisitMut,
};
use wit_bindgen_core::{
    WorldGenerator,
    wit_parser::{PackageId, Resolve, UnresolvedPackageGroup},
};
use wit_bindgen_rust::{Opts, WithOption};

use crate::{fpi, manifest_paths};

#[derive(Default)]
struct GenerateArgs {
    inline: Option<LitStr>,
    /// Custom `with` entries parsed from the macro input.
    /// Each entry maps a WIT interface/type to either `generate` or a Rust path.
    /// Stored directly as `(String, WithOption)` to avoid an intermediate representation.
    with_entries: Vec<(String, WithOption)>,
}

/// Parses a single `with` entry like `"miden:foo/bar": generate` or `"miden:foo/bar": ::my::Path`.
fn parse_with_entry(input: ParseStream<'_>) -> syn::Result<(String, WithOption)> {
    let key: LitStr = input.parse()?;
    input.parse::<Token![:]>()?;
    let path: syn::Path = input.parse()?;

    // Check if the path is the special `generate` keyword
    let option = if path.leading_colon.is_none()
        && path.segments.len() == 1
        && path.segments.first().is_some_and(|seg| seg.ident == "generate")
    {
        WithOption::Generate
    } else {
        // Convert syn::Path to string, removing spaces for consistency
        let path_str = path.to_token_stream().to_string().replace(' ', "");
        WithOption::Path(path_str)
    };

    Ok((key.value(), option))
}

impl Parse for GenerateArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut args = GenerateArgs::default();

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            let name = ident.to_string();
            input.parse::<Token![=]>()?;

            if name == "inline" {
                if args.inline.is_some() {
                    return Err(syn::Error::new(ident.span(), "duplicate `inline` argument"));
                }
                args.inline = Some(input.parse()?);
            } else if name == "with" {
                if !args.with_entries.is_empty() {
                    return Err(syn::Error::new(ident.span(), "duplicate `with` argument"));
                }
                let content;
                syn::braced!(content in input);
                // Parse comma-separated with entries directly into (String, WithOption) pairs
                while !content.is_empty() {
                    args.with_entries.push(parse_with_entry(&content)?);
                    if content.peek(Token![,]) {
                        content.parse::<Token![,]>()?;
                    }
                }
            } else {
                return Err(syn::Error::new(
                    ident.span(),
                    format!("unsupported generate! argument `{name}`"),
                ));
            }

            if input.peek(Token![,]) {
                let _ = input.parse::<Token![,]>()?;
            }
        }

        Ok(args)
    }
}

/// Implements the expansion logic for the `generate!` macro.
pub(crate) fn expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input_tokens: proc_macro2::TokenStream = input.into();
    let args = if input_tokens.is_empty() {
        GenerateArgs::default()
    } else {
        match syn::parse2::<GenerateArgs>(input_tokens) {
            Ok(parsed) => parsed,
            Err(err) => return err.to_compile_error().into(),
        }
    };

    let resolve_opts = manifest_paths::ResolveOptions {
        allow_missing_local_wit: args.inline.is_some(),
    };

    match manifest_paths::resolve_wit_paths(resolve_opts) {
        Ok(config) => {
            if config.paths.is_empty() {
                return Error::new(
                    Span::call_site(),
                    "no WIT dependencies declared under \
                     [package.metadata.component.target.dependencies]",
                )
                .to_compile_error()
                .into();
            }

            let inline_world = args
                .inline
                .as_ref()
                .and_then(|src| manifest_paths::extract_world_name(&src.value()));
            let world_value = inline_world.or_else(|| config.world.clone());

            if args.inline.is_some() && world_value.is_none() {
                return Error::new(
                    Span::call_site(),
                    "failed to detect world name for inline WIT provided to generate!",
                )
                .to_compile_error()
                .into();
            }

            match generate_bindings(&args, &config, world_value.as_deref()) {
                Ok(raw_bindings) => quote! {
                    // Wrap the bindings in the `bindings` module since `generate!` makes a top level
                    // module named after the package namespace which is `miden` for all our projects
                    // so it conflicts with the `miden` crate (SDK)
                    #[doc(hidden)]
                    #[allow(dead_code)]
                    pub mod bindings {
                        #raw_bindings
                    }
                }
                .into(),
                Err(err) => err.to_compile_error().into(),
            }
        }
        Err(err) => err.to_compile_error().into(),
    }
}

/// Generates WIT bindings using `wit-bindgen` directly instead of the `generate!` macro.
///
/// The `world` parameter specifies which world to generate bindings for. This should already
/// be resolved by the caller (either from inline WIT or from the local wit/ directory).
/// If `None`, wit-bindgen will attempt to select a default world from the loaded packages.
fn generate_bindings(
    args: &GenerateArgs,
    config: &manifest_paths::ResolvedWit,
    world: Option<&str>,
) -> Result<TokenStream2, Error> {
    generate_bindings_from_sources(
        &config.paths,
        args.inline.as_ref().map(|src| src.value()).as_deref(),
        world,
        &args.with_entries,
        &[],
    )
}

/// Generates inline WIT bindings and injects FPI imports for the selected dependency interfaces.
pub(crate) fn generate_inline_fpi_bindings(
    config: &manifest_paths::ResolvedWit,
    inline_source: &str,
    world: &str,
    fpi_imports: &[String],
    with_entries: &[(String, WithOption)],
) -> Result<TokenStream2, Error> {
    generate_bindings_from_sources(
        &config.paths,
        Some(inline_source),
        Some(world),
        with_entries,
        fpi_imports,
    )
}

/// Generates inline bindings for an import-only world without injecting FPI variants.
///
/// Used by the `#[component]` sibling generator: the imported dependency functions are kept
/// as-is and lower to direct cross-context calls, so no `fpi-*` companions are synthesized.
pub(crate) fn generate_inline_import_bindings(
    config: &manifest_paths::ResolvedWit,
    inline_source: &str,
    world: &str,
    with_entries: &[(String, WithOption)],
) -> Result<TokenStream2, Error> {
    generate_bindings_from_sources(
        &config.paths,
        Some(inline_source),
        Some(world),
        with_entries,
        &[],
    )
}

/// Generates WIT bindings from resolved source paths and optional inline source.
fn generate_bindings_from_sources(
    paths: &[String],
    inline_source: Option<&str>,
    world: Option<&str>,
    with_entries: &[(String, WithOption)],
    fpi_imports: &[String],
) -> Result<TokenStream2, Error> {
    let mut wit_sources = load_wit_sources(paths, inline_source)?;

    let world_id = wit_sources
        .resolve
        .select_world(&wit_sources.packages, world)
        .map_err(|err| Error::new(Span::call_site(), err.to_string()))?;
    fpi::inject_imports(&mut wit_sources.resolve, world_id, fpi_imports)?;

    let mut opts = Opts {
        generate_all: true,
        runtime_path: Some("::miden::wit_bindgen::rt".to_string()),
        default_bindings_module: Some("bindings".to_string()),
        ..Opts::default()
    };
    push_custom_with_entries(&mut opts, with_entries);
    push_default_with_entries(&mut opts);

    let mut generated_files = wit_bindgen_core::Files::default();
    let mut generator = opts.build();
    generator
        .generate(&mut wit_sources.resolve, world_id, &mut generated_files)
        .map_err(|err| Error::new(Span::call_site(), err.to_string()))?;

    let (_, src_bytes) = generated_files
        .iter()
        .next()
        .ok_or_else(|| Error::new(Span::call_site(), "wit-bindgen emitted no bindings"))?;
    let src = std::str::from_utf8(src_bytes)
        .map_err(|err| Error::new(Span::call_site(), format!("invalid UTF-8: {err}")))?;
    let mut tokens: TokenStream2 = src
        .parse()
        .map_err(|err| Error::new(Span::call_site(), format!("failed to parse bindings: {err}")))?;

    // Include a dummy `include_bytes!` for any files we read so rustc knows that
    // we depend on the contents of those files.
    for path in wit_sources.files_read {
        let utf8_path = path.to_str().ok_or_else(|| {
            Error::new(
                Span::call_site(),
                format!("path '{}' contains invalid UTF-8", path.display()),
            )
        })?;
        tokens.extend(quote! {
            const _: &[u8] = include_bytes!(#utf8_path);
        });
    }

    Ok(tokens)
}

/// Result of loading and parsing WIT sources from file paths and optional inline content.
struct LoadedWitSources {
    /// The resolved WIT definitions containing all types, interfaces, and worlds.
    resolve: Resolve,
    /// Package IDs to use for world selection. When inline source is provided, this contains
    /// only the inline package; otherwise it contains all packages from file paths.
    packages: Vec<PackageId>,
    /// File paths that were read during WIT parsing. Used to generate dummy `include_bytes!`
    /// calls so rustc knows to recompile when these files change.
    files_read: Vec<PathBuf>,
}

/// Loads WIT sources from file paths and optionally an inline source.
fn load_wit_sources(
    paths: &[String],
    inline_source: Option<&str>,
) -> Result<LoadedWitSources, Error> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").map_err(|err| {
        Error::new(Span::call_site(), format!("failed to read CARGO_MANIFEST_DIR: {err}"))
    })?;
    let manifest_dir = PathBuf::from(manifest_dir);

    let mut resolve = Resolve::default();
    let mut packages = Vec::new();
    let mut files = Vec::new();

    // Load WIT definitions from file paths. These are always loaded to populate the resolver
    // with type definitions that the inline source may depend on.
    for path in paths {
        let path_buf = PathBuf::from(path);
        let absolute = if path_buf.is_absolute() {
            path_buf
        } else {
            manifest_dir.join(path_buf)
        };
        let normalized = fs::canonicalize(&absolute).unwrap_or(absolute);
        let (pkg, sources) = resolve.push_path(normalized.clone()).map_err(|err| {
            Error::new(
                Span::call_site(),
                format!("failed to load WIT from '{}': {err}", normalized.display()),
            )
        })?;
        packages.push(pkg);
        files.extend(sources.paths().map(|p| p.to_owned()));
    }

    if let Some(src) = inline_source {
        // When inline source is provided, it becomes the primary package for world selection.
        // We clear previously collected package IDs because the inline source defines the world
        // we want to generate bindings for. The file-based packages are still loaded above and
        // remain in the resolver - they provide type definitions that the inline world imports.
        packages.clear();
        let group = UnresolvedPackageGroup::parse("inline", src)
            .map_err(|err| Error::new(Span::call_site(), err.to_string()))?;
        let pkg = resolve
            .push_group(group)
            .map_err(|err| Error::new(Span::call_site(), err.to_string()))?;
        packages.push(pkg);
    }

    Ok(LoadedWitSources {
        resolve,
        packages,
        files_read: files,
    })
}

/// Pushes user-provided `with` entries to the wit-bindgen options.
fn push_custom_with_entries(opts: &mut Opts, entries: &[(String, WithOption)]) {
    opts.with.extend(entries.iter().cloned());
}

/// Pushes default `with` entries that map Miden base types to SDK types.
fn push_default_with_entries(opts: &mut Opts) {
    opts.with
        .push(("miden:base/core-types@1.0.0".to_string(), WithOption::Generate));
    push_path_entry(opts, "miden:base/core-types@1.0.0/felt", "::miden::Felt");
    push_path_entry(opts, "miden:base/core-types@1.0.0/word", "::miden::Word");
    push_path_entry(opts, "miden:base/core-types@1.0.0/asset", "::miden::Asset");
    push_path_entry(opts, "miden:base/core-types@1.0.0/account-id", "::miden::AccountId");
    push_path_entry(opts, "miden:base/core-types@1.0.0/tag", "::miden::Tag");
    push_path_entry(opts, "miden:base/core-types@1.0.0/note-type", "::miden::NoteType");
    push_path_entry(opts, "miden:base/core-types@1.0.0/recipient", "::miden::Recipient");
    push_path_entry(opts, "miden:base/core-types@1.0.0/note-idx", "::miden::NoteIdx");
}

fn push_path_entry(opts: &mut Opts, key: &str, value: &str) {
    opts.with.push((key.to_string(), WithOption::Path(value.to_string())));
}

/// Qualifies type paths in a function signature with the module path prefix.
///
/// This transforms simple type names (e.g., `StructA`) into fully qualified paths
/// (e.g., `miden::component::component::StructA`) so they resolve correctly when
/// the method is placed at the bindings root level.
pub(crate) fn qualify_signature_types(sig: &mut syn::Signature, module_path: &[syn::Ident]) {
    struct TypeQualifier<'a> {
        module_path: &'a [syn::Ident],
    }

    impl VisitMut for TypeQualifier<'_> {
        fn visit_type_path_mut(&mut self, type_path: &mut TypePath) {
            // Only qualify paths that:
            // 1. Don't already have a leading colon (not absolute like `::foo`)
            // 2. Are simple single-segment paths (like `StructA`, not `foo::Bar`)
            // 3. Don't start with common primitive/std type names
            if type_path.qself.is_none()
                && type_path.path.leading_colon.is_none()
                && type_path.path.segments.len() == 1
            {
                let first_segment = &type_path.path.segments[0].ident;
                let name = first_segment.to_string();

                // Skip primitive types and common std types
                if is_primitive_or_std_type(&name) {
                    return;
                }

                // Build the qualified path: module_path::TypeName
                let mut new_segments = syn::punctuated::Punctuated::new();
                for ident in self.module_path {
                    new_segments.push(syn::PathSegment {
                        ident: ident.clone(),
                        arguments: syn::PathArguments::None,
                    });
                }
                // Add the original type segment (preserving generics)
                new_segments.push(type_path.path.segments[0].clone());

                type_path.path.segments = new_segments;
            }

            // Continue visiting nested types (e.g., generics)
            syn::visit_mut::visit_type_path_mut(self, type_path);
        }
    }

    let mut qualifier = TypeQualifier { module_path };
    qualifier.visit_signature_mut(sig);
}

/// Returns true if the name is a primitive type or common std type that shouldn't be qualified.
///
/// This list covers Rust primitives and common standard library types. WIT-generated bindings
/// only use a subset of these (primitives, String, Vec, Option, Result), but we include
/// additional common types for safety. Types like `Rc`, `Arc`, `RefCell` are not used by
/// wit-bindgen and are intentionally omitted.
fn is_primitive_or_std_type(name: &str) -> bool {
    matches!(
        name,
        "bool"
            | "char"
            | "str"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "f32"
            | "f64"
            | "String"
            | "Vec"
            | "Option"
            | "Result"
            | "Self"
    )
}

/// Extracts argument identifiers from a function signature.
///
/// Returns an error if the function contains a receiver (`self`) or uses
/// unsupported argument patterns (e.g., destructuring patterns).
pub(crate) fn collect_arg_idents(func: &ItemFn) -> syn::Result<Vec<syn::Ident>> {
    func.sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(_) => {
                Err(Error::new(func.sig.ident.span(), "unexpected receiver in generated function"))
            }
            FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                Pat::Ident(pat_ident) => Ok(pat_ident.ident.clone()),
                other => Err(Error::new(
                    other.span(),
                    format!(
                        "unsupported argument pattern `{}` in generated function",
                        quote!(#other)
                    ),
                )),
            },
        })
        .collect()
}

/// Determines whether a wrapper struct should be generated for the given module.
///
/// Returns `false` for:
/// - Empty paths
/// - `exports` modules (these are user-implemented exports, not imports)
/// - Modules starting with underscore (internal/private modules)
/// - Non-leaf modules (modules that contain nested modules)
pub(crate) fn should_generate_struct(path: &[syn::Ident], items: &[Item]) -> bool {
    if path.is_empty() {
        return false;
    }
    let first = path[0].to_string();
    if first == "exports" {
        return false;
    }
    if first.starts_with('_') {
        return false;
    }
    let last = path.last().unwrap().to_string();
    if last.starts_with('_') {
        return false;
    }
    // Only generate for leaf modules (no nested modules)
    !items.iter().any(|item| matches!(item, Item::Mod(_)))
}

/// Formats a module path as a `::` separated string for use in documentation.
pub(crate) fn format_module_path(path: &[syn::Ident]) -> String {
    path.iter().map(|ident| ident.to_string()).collect::<Vec<_>>().join("::")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_generate_struct_empty_path() {
        let empty_items: Vec<Item> = vec![];
        assert!(!should_generate_struct(&[], &empty_items));
    }

    #[test]
    fn test_should_generate_struct_exports_excluded() {
        let empty_items: Vec<Item> = vec![];
        let path = vec![syn::Ident::new("exports", Span::call_site())];
        assert!(!should_generate_struct(&path, &empty_items));

        let path = vec![
            syn::Ident::new("exports", Span::call_site()),
            syn::Ident::new("foo", Span::call_site()),
        ];
        assert!(!should_generate_struct(&path, &empty_items));
    }

    #[test]
    fn test_should_generate_struct_underscore_excluded() {
        let empty_items: Vec<Item> = vec![];
        let path = vec![syn::Ident::new("_private", Span::call_site())];
        assert!(!should_generate_struct(&path, &empty_items));

        let path = vec![
            syn::Ident::new("miden", Span::call_site()),
            syn::Ident::new("_internal", Span::call_site()),
        ];
        assert!(!should_generate_struct(&path, &empty_items));
    }

    #[test]
    fn test_should_generate_struct_valid_leaf_modules() {
        let empty_items: Vec<Item> = vec![];
        let path = vec![syn::Ident::new("miden", Span::call_site())];
        assert!(should_generate_struct(&path, &empty_items));

        let path = vec![
            syn::Ident::new("miden", Span::call_site()),
            syn::Ident::new("basic_wallet", Span::call_site()),
        ];
        assert!(should_generate_struct(&path, &empty_items));
    }

    #[test]
    fn test_should_generate_struct_non_leaf_excluded() {
        let path = vec![syn::Ident::new("miden", Span::call_site())];
        // Items containing a nested module
        let items_with_mod: Vec<Item> = vec![syn::parse_quote! { mod nested {} }];
        assert!(!should_generate_struct(&path, &items_with_mod));

        // Items with only functions (leaf module) should be allowed
        let items_with_fn: Vec<Item> = vec![syn::parse_quote! { pub fn foo() {} }];
        assert!(should_generate_struct(&path, &items_with_fn));
    }

    #[test]
    fn test_format_module_path() {
        let path = vec![
            syn::Ident::new("miden", Span::call_site()),
            syn::Ident::new("basic_wallet", Span::call_site()),
        ];
        assert_eq!(format_module_path(&path), "miden::basic_wallet");
    }

    #[test]
    fn test_format_module_path_empty() {
        assert_eq!(format_module_path(&[]), "");
    }

    #[test]
    fn test_collect_arg_idents() {
        let func: ItemFn = syn::parse_quote! {
            pub fn foo(a: u32, b: String, c: Vec<u8>) {}
        };
        let idents = collect_arg_idents(&func).unwrap();
        let names: Vec<_> = idents.iter().map(|i| i.to_string()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_collect_arg_idents_empty() {
        let func: ItemFn = syn::parse_quote! {
            pub fn no_args() {}
        };
        let idents = collect_arg_idents(&func).unwrap();
        assert!(idents.is_empty());
    }

    #[test]
    fn test_qualify_signature_types() {
        let mut sig: syn::Signature = syn::parse_quote! {
            fn test_fn(a: StructA, b: u64) -> StructB
        };
        let path = vec![
            syn::Ident::new("miden", Span::call_site()),
            syn::Ident::new("component", Span::call_site()),
        ];
        qualify_signature_types(&mut sig, &path);

        // Check that the custom types are qualified with the module path
        let sig_str = sig.to_token_stream().to_string();
        assert!(sig_str.contains("miden :: component :: StructA"));
        assert!(sig_str.contains("miden :: component :: StructB"));
        // Primitives should not be qualified
        assert!(sig_str.contains("u64"));
        assert!(!sig_str.contains("miden :: component :: u64"));
    }

    #[test]
    fn test_parse_with_entry_generate() {
        let input: TokenStream2 = quote! { "miden:foo/bar": generate };
        let parsed = syn::parse2::<GenerateArgs>(quote! { with = { #input } }).unwrap();

        assert_eq!(parsed.with_entries.len(), 1);
        assert_eq!(parsed.with_entries[0].0, "miden:foo/bar");
        assert!(matches!(parsed.with_entries[0].1, WithOption::Generate));
    }

    #[test]
    fn test_parse_with_entry_path() {
        let input: TokenStream2 = quote! { "miden:foo/bar": ::my::custom::Type };
        let parsed = syn::parse2::<GenerateArgs>(quote! { with = { #input } }).unwrap();

        assert_eq!(parsed.with_entries.len(), 1);
        assert_eq!(parsed.with_entries[0].0, "miden:foo/bar");
        match &parsed.with_entries[0].1 {
            WithOption::Path(p) => assert_eq!(p, "::my::custom::Type"),
            _ => panic!("expected Path variant"),
        }
    }

    #[test]
    fn test_parse_multiple_with_entries() {
        let parsed = syn::parse2::<GenerateArgs>(quote! {
            with = {
                "miden:a/b": generate,
                "miden:c/d": ::foo::Bar
            }
        })
        .unwrap();

        assert_eq!(parsed.with_entries.len(), 2);
        assert_eq!(parsed.with_entries[0].0, "miden:a/b");
        assert_eq!(parsed.with_entries[1].0, "miden:c/d");
    }
}
