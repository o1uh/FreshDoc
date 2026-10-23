// FD-39 фаза 1: извлечение функций и godoc из Go-файлов.
//
// Использование:
//	gominer -dir <каталог с .go файлами>   (рекурсивно, без vendor/, без *_test.go)
//
// Вывод в stdout: JSON-массив объектов {file, symbol, sig, doc, code}.
// file — путь относительно -dir; symbol — Name или Recv.Name для методов;
// sig — заголовок "func ..." без тела; doc — текст godoc без "//";
// code — полный текст объявления через go/printer (БЕЗ godoc: дифф строится
// только по коду, дока — отдельно, иначе судья видел бы правку доки в диффе).
//
// Только стандартная библиотека (go/parser, go/ast, go/printer).
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Func struct {
	File   string `json:"file"`
	Symbol string `json:"symbol"`
	Sig    string `json:"sig"`
	Doc    string `json:"doc"`
	Code   string `json:"code"`
}

func recvName(fl *ast.FieldList) string {
	if fl == nil || len(fl.List) == 0 {
		return ""
	}
	t := fl.List[0].Type
	switch x := t.(type) {
	case *ast.StarExpr:
		if id, ok := x.X.(*ast.Ident); ok {
			return id.Name
		}
	case *ast.Ident:
		return x.Name
	}
	return ""
}

func render(fset *token.FileSet, node any) string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, node); err != nil {
		return ""
	}
	return buf.String()
}

// sigOf строит заголовок "func Name(params) results" / "func (r T) Name(...)".
// recvStr форматирует ресивер вручную: go/printer не умеет печатать
// *ast.Field / *ast.FieldList отдельно ("unsupported node type"),
// а тип-ресивер (Expr) печатается нормально.
func recvStr(fset *token.FileSet, fl *ast.FieldList) string {
	if fl == nil || len(fl.List) == 0 {
		return ""
	}
	f := fl.List[0]
	t := render(fset, f.Type)
	var names []string
	for _, n := range f.Names {
		names = append(names, n.Name)
	}
	if len(names) > 0 {
		return strings.Join(names, ", ") + " " + t
	}
	return t
}

func sigOf(fset *token.FileSet, d *ast.FuncDecl) string {
	ft := render(fset, d.Type)
	if r := recvStr(fset, d.Recv); r != "" {
		return "func (" + r + ") " + d.Name.Name + strings.TrimPrefix(ft, "func")
	}
	return "func " + d.Name.Name + strings.TrimPrefix(ft, "func")
}

func parseFile(fset *token.FileSet, path, rel string, out *[]Func) {
	f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warn: skip %s: %v\n", rel, err)
		return
	}
	for _, decl := range f.Decls {
		d, ok := decl.(*ast.FuncDecl)
		if !ok || d.Name == nil {
			continue
		}
		sym := d.Name.Name
		if r := recvName(d.Recv); r != "" {
			sym = r + "." + sym
		}
		doc := ""
		if d.Doc != nil {
			doc = strings.TrimSpace(d.Doc.Text())
		}
		d.Doc = nil // код без godoc: дифф строится только по коду (R1), дока — отдельно
		*out = append(*out, Func{
			File:   rel,
			Symbol: sym,
			Sig:    sigOf(fset, d),
			Doc:    doc,
			Code:   render(fset, d),
		})
	}
}

func main() {
	dir := flag.String("dir", "", "каталог с .go файлами (рекурсивно)")
	flag.Parse()
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "нужен -dir")
		os.Exit(2)
	}
	fset := token.NewFileSet()
	var out []Func
	err := filepath.Walk(*dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			if info.Name() == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		rel, _ := filepath.Rel(*dir, path)
		parseFile(fset, path, filepath.ToSlash(rel), &out)
		return nil
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		return out[i].Symbol < out[j].Symbol
	})
	if out == nil {
		out = []Func{}
	}
	json.NewEncoder(os.Stdout).Encode(out)
}
