"""
Generate a large multi-language dataset (AI-like and human-like) across languages:
- Python, JavaScript, Java, C++, C#, Go, Rust

Writes files to:
  data/train/ai/
  data/train/human/

Default target: ~6,020 files total (per_lang_per_class * 7 * 2).
"""

from pathlib import Path
import textwrap


def main(per_lang_per_class: int = 430) -> None:
    root = Path('data/train')
    ai_dir = root / 'ai'
    human_dir = root / 'human'
    ai_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)

    def ext_for(lang: str) -> str:
        return 'cs' if lang == 'cs' else ('cpp' if lang == 'cpp' else lang)

    langs = {
        'py': {
            'human': lambda i: textwrap.dedent(f'''\
                def f{i}(n):
                    a,b=0,1
                    out=[]
                    for _ in range(n):
                        out.append(a)
                        a,b=b,a+b
                    return out
            '''),
            'ai': lambda i: textwrap.dedent(f'''\
                """Deterministic Fibonacci sequence generator (verbose)."""
                from typing import List
                def generate_fibonacci_sequence_{i}(count: int) -> List[int]:
                    prev:int=0; cur:int=1; seq:List[int]=[]
                    for _ in range(count):
                        seq.append(prev)
                        nxt:int=prev+cur
                        prev=cur; cur=nxt
                    return seq
            '''),
        },
        'js': {
            'human': lambda i: textwrap.dedent(f'''\
                function uniq{i}(arr) {{
                  const s = new Set();
                  const out = [];
                  for (const x of arr) if (!s.has(x)) {{ s.add(x); out.push(x); }}
                  return out;
                }}
            '''),
            'ai': lambda i: textwrap.dedent(f'''\
                /** Stable unique extraction (verbose identifiers). */
                export function deriveUniqueSequence{i}(values) {{
                  const encountered = new Set();
                  const output = [];
                  for (const value of values) {{
                    if (!encountered.has(value)) {{
                      encountered.add(value);
                      output.push(value);
                    }}
                  }}
                  return output;
                }}
            '''),
        },
        'java': {
            'human': lambda i: textwrap.dedent(f'''\
                public class Sum{i} {{
                  public static int sum(int[] a) {{
                    int s=0; for (int v: a) s+=v; return s;
                  }}
                }}
            '''),
            'ai': lambda i: textwrap.dedent(f'''\
                /** Deterministic summation with explicit semantics. */
                public class DeterministicSum{i} {{
                  public static int compute(int[] inputArray) {{
                    int cumulative = 0;
                    for (int idx = 0; idx < inputArray.length; idx++) {{
                      cumulative = cumulative + inputArray[idx];
                    }}
                    return cumulative;
                  }}
                }}
            '''),
        },
        'cpp': {
            'human': lambda i: textwrap.dedent(f'''\
                #include <vector>
                int sum{i}(const std::vector<int>& a) {{
                  int s=0; for (int v: a) s+=v; return s;
                }}
            '''),
            'ai': lambda i: textwrap.dedent(f'''\
                #include <vector>
                // Accumulate values with explicit iteration.
                int compute_sum_{i}(const std::vector<int>& input) {{
                  int cumulative = 0;
                  for (size_t index = 0; index < input.size(); ++index) {{
                    cumulative = cumulative + input[index];
                  }}
                  return cumulative;
                }}
            '''),
        },
        'cs': {
            'human': lambda i: textwrap.dedent(f'''\
                public static class U{i} {{
                  public static int Sum(int[] a) {{
                    int s=0; foreach (var v in a) s+=v; return s;
                  }}
                }}
            '''),
            'ai': lambda i: textwrap.dedent(f'''\
                /// Compute summation with explicit loop semantics.
                public static class DeterministicSum{i} {{
                  public static int Compute(int[] inputArray) {{
                    int cumulative = 0;
                    for (int index = 0; index < inputArray.Length; index++) {{
                      cumulative = cumulative + inputArray[index];
                    }}
                    return cumulative;
                  }}
                }}
            '''),
        },
        'go': {
            'human': lambda i: textwrap.dedent(f'''\
                package m{i}
                func Sum(a []int) int {{
                  s:=0
                  for _,v:= range a {{ s+=v }}
                  return s
                }}
            '''),
            'ai': lambda i: textwrap.dedent(f'''\
                package m{i}
                // Deterministic summation with explicit index semantics.
                func ComputeSum(input []int) int {{
                  cum := 0
                  for idx:=0; idx < len(input); idx++ {{
                    cum = cum + input[idx]
                  }}
                  return cum
                }}
            '''),
        },
        'rs': {
            'human': lambda i: textwrap.dedent(f'''\
                pub fn sum_{i}(a: &[i32]) -> i32 {{
                    let mut s = 0;
                    for v in a {{ s += *v; }}
                    s
                }}
            '''),
            'ai': lambda i: textwrap.dedent(f'''\
                /// Deterministic summation with explicit iteration semantics.
                pub fn compute_sum_{i}(input: &[i32]) -> i32 {{
                    let mut cumulative: i32 = 0;
                    for idx in 0..input.len() {{
                        cumulative = cumulative + input[idx];
                    }}
                    cumulative
                }}
            '''),
        },
    }

    total = 0
    for lang, gens in langs.items():
        extension = ext_for(lang)
        for i in range(per_lang_per_class):
            (human_dir / f"{lang}_human_ext_{i:04d}.{extension}").write_text(gens['human'](i), encoding='utf-8')
            (ai_dir / f"{lang}_ai_ext_{i:04d}.{extension}").write_text(gens['ai'](i), encoding='utf-8')
            total += 2

    print(f"Generated {total} files under {root}")


if __name__ == '__main__':
    main()


