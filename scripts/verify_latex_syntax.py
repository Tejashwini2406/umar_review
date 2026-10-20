import os
import re
import sys

def expand_tex(filepath, root_dir=None):
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(filepath))

    if not os.path.exists(filepath):
        if not filepath.endswith('.tex') and os.path.exists(filepath + '.tex'):
            filepath += '.tex'
        else:
            return f"% Failed to load {filepath}\n"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    full_content = []
    for line in lines:
        match = re.search(r'\\input\{([^}]+)\}', line)
        if match:
            sub_path = match.group(1)
            # Try relative to root directory first (standard LaTeX behavior)
            cand1 = os.path.join(root_dir, sub_path)
            cand2 = os.path.join(os.path.dirname(filepath), sub_path)
            
            if os.path.exists(cand1) or os.path.exists(cand1 + '.tex'):
                full_content.append(expand_tex(cand1, root_dir))
            elif os.path.exists(cand2) or os.path.exists(cand2 + '.tex'):
                full_content.append(expand_tex(cand2, root_dir))
            else:
                full_content.append(expand_tex(cand1, root_dir))
        else:
            full_content.append(line)
            
    return "".join(full_content)

def verify_latex(filepath, bibpath=None):
    if not os.path.exists(filepath):
        print(f"[ERROR] Target file does not exist: {filepath}")
        return False
        
    root_dir = os.path.dirname(os.path.abspath(filepath))
    content = expand_tex(filepath, root_dir)

    errors = []
    warnings = []

    lines = content.split('\n')

    # 1. Environment stack check
    env_stack = []
    for i, line in enumerate(lines, 1):
        clean_line = line.split('%')[0] if '%' in line and (line.find('%') == 0 or line[line.find('%')-1] != '\\') else line

        begins = re.findall(r'\\begin\{([^}]+)\}', clean_line)
        ends = re.findall(r'\\end\{([^}]+)\}', clean_line)

        for b in begins:
            env_stack.append((b, i))
        for e in ends:
            if not env_stack:
                errors.append(f"Line {i}: Found \\end{{{e}}} without matching \\begin")
            else:
                last_b, last_line = env_stack.pop()
                if last_b != e:
                    errors.append(f"Line {i}: Mismatched environment \\end{{{e}}}, expected \\end{{{last_b}}} (opened line {last_line})")

    if env_stack:
        for b, line_num in env_stack:
            errors.append(f"Line {line_num}: Unclosed environment \\begin{{{b}}}")

    # 2. Check captions for unescaped %
    for i, line in enumerate(lines, 1):
        if r'\caption{' in line or r'\caption[' in line:
            cap_text = line.replace(r'\%', '')
            if '%' in cap_text and (cap_text.find('%') == 0 or cap_text[cap_text.find('%')-1] != '\\'):
                errors.append(f"Line {i}: Fatal unescaped '%' in caption: {line.strip()}")

    # 3. Check for unescaped ampersands outside tabular/matrix/align
    in_tabular = False
    for i, line in enumerate(lines, 1):
        clean_line = line.split('%')[0]
        if any(env in clean_line for env in [r'\begin{tabular}', r'\begin{tabular*}', r'\begin{align}', r'\begin{matrix}', r'\begin{bmatrix}', r'\begin{cases}']):
            in_tabular = True

        if not in_tabular:
            clean_text = clean_line.replace(r'\&', '')
            if '&' in clean_text:
                errors.append(f"Line {i}: Unescaped '&' outside tabular/align: '{line.strip()}'")

        if any(env in clean_line for env in [r'\end{tabular}', r'\end{tabular*}', r'\end{align}', r'\end{matrix}', r'\end{bmatrix}', r'\end{cases}']):
            in_tabular = False

    # 4. Check for unescaped underscores outside math mode and command args
    text_only = content
    text_only = re.sub(r'\\begin\{(equation|align|matrix)\}.*?\\end\{\1\}', '', text_only, flags=re.DOTALL)
    text_only = re.sub(r'\$[^\$]+\$', '', text_only)
    text_only = re.sub(r'\\(includegraphics|label|ref|cite|texttt|url|input|bibliography|bibliographystyle|author|IEEEauthorblockN|IEEEauthorblockA)(\[[^\]]*\])?\{[^}]*\}', '', text_only)

    for i, line in enumerate(text_only.split('\n'), 1):
        clean_line = line.split('%')[0]
        if '`' in clean_line:
            warnings.append(f"Line {i}: Markdown backtick found: {clean_line.strip()}")
        
        clean_text = clean_line.replace(r'\_', '')
        if '_' in clean_text:
            errors.append(f"Line {i}: Illegal unescaped '_' in text mode: '{clean_line.strip()}'")

    # 5. Check labels vs refs
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', content))

    missing_labels = refs - labels
    if missing_labels:
        for ml in missing_labels:
            errors.append(f"Undefined reference label: \\ref{{{ml}}}")

    # 6. Check citations
    cites = set()
    for match in re.findall(r'\\cite\{([^}]+)\}', content):
        for c in match.split(','):
            cites.add(c.strip())

    bib_keys = set()
    if bibpath and os.path.exists(bibpath):
        with open(bibpath, 'r', encoding='utf-8') as bf:
            bib_text = bf.read()
            bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib_text))

    bibitem_keys = set(re.findall(r'\\bibitem\{([^}]+)\}', content))
    all_known_keys = bib_keys.union(bibitem_keys)

    missing_cites = cites - all_known_keys
    if missing_cites:
        for mc in missing_cites:
            errors.append(f"Undefined citation key: \\cite{{{mc}}}")

    # 7. Check graphic inputs
    graphics = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content)
    for g in graphics:
        g_path = os.path.join(root_dir, g)
        if not os.path.exists(g_path) and not any(os.path.exists(g_path + ext) for ext in ['.pdf', '.png', '.jpg', '.eps']):
            errors.append(f"Missing graphic asset: {g}")

    print(f"=== FULL LATEX SYNTAX AUDIT RESULTS ({filepath}) ===")
    print(f"Total Fatal Errors Found: {len(errors)}")
    for err in errors:
        print(f"  [ERROR] {err}")

    print(f"Total Warnings Found: {len(warnings)}")
    for warn in warnings:
        print(f"  [WARN]  {warn}")

    return len(errors) == 0

if __name__ == "__main__":
    tex_path = sys.argv[1] if len(sys.argv) > 1 else "ieee_paper_draft/main.tex"
    bib_path = sys.argv[2] if len(sys.argv) > 2 else "ieee_paper_draft/references.bib"
    success = verify_latex(tex_path, bib_path)
    if not success:
        sys.exit(1)
