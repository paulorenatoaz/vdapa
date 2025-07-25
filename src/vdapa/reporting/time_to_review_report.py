import math
from collections import Counter
from pathlib import Path
import json

import pandas as pd
import matplotlib
from dateutil.utils import today

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from vdapa.config import config, BASE_DIR
from vdapa.utils import setup_logging

logger = setup_logging("reporting", "time_to_review_report")

# CDN Links
DATA_TABLES_CSS = "https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css"
SCROLLER_CSS    = "https://cdn.datatables.net/scroller/2.0.7/css/dataTables.scroller.min.css"

JQUERY_JS       = "https://code.jquery.com/jquery-3.6.0.min.js"
DATA_TABLES_JS  = "https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"
SCROLLER_JS     = "https://cdn.datatables.net/scroller/2.0.7/js/dataTables.scroller.min.js"


def get_latest_model_timestamp(train_summary_path):
    with open(train_summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    if not summary:
        logger.error("Empty train summary: %s", train_summary_path)
        raise ValueError(f"Empty train summary at {train_summary_path}")
    return summary[0].get('timestamp')


def load_predictions(model_timestamp):
    preds_dir = BASE_DIR / 'data' / 'processed' / 'time_to_review_model'
    pred_file = preds_dir / f'prediction_{model_timestamp}.json'
    if not pred_file.exists():
        logger.error("Prediction file missing: %s", pred_file)
        raise FileNotFoundError(f"Missing predictions: {pred_file}")
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[0].get('prediction', [])


def load_normalized_advisories_df():
    adv_file = BASE_DIR / 'data' / 'processed' / 'normalized_advisory.json'
    records = []
    with open(adv_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping bad JSON line in advisories")
                continue
            records.append(rec)
    df = pd.DataFrame(records)
    # Parse dates
    date_cols = ['published_at', 'modified_at', 'github_reviewed_at', 'nvd_published_at', 'withdrawn_at']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # Numeric fields
    for col in ['time_to_review', 'time_to_nvd', 'cvss_number']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Derived counts
    df['num_cwes'] = df.get('cwe_ids', []).apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_refs'] = df.get('references', []).apply(lambda x: len(x) if isinstance(x, list) else 0)
    return df


def generate_histogram(df, base_name, var_name, xlabel, output_dir):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_file = out / f'{base_name}_{var_name}_histogram.png'

    data = df[var_name].dropna()

    plt.figure()


    if pd.api.types.is_numeric_dtype(data):
        if data.nunique() <= 20:
            bins = range(int(data.min()), int(data.max()) + 2)
        else:
            bins = 'auto'
        plt.hist(data, bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
    else:
        if pd.api.types.is_object_dtype(data) and isinstance(data.iloc[0], list):
            # If it's a list of categories, flatten
            data = data.explode().dropna()


        top_categories = data.value_counts().nlargest(10).index
        filtered = data.where(data.isin(top_categories), other='Other')
        counts = filtered.value_counts()

        if len(top_categories) > 9:
            plt.barh(counts.index, counts.values)
            plt.ylabel(xlabel)
            plt.xlabel('Count')
        else:
            plt.bar(counts.index, counts.values)
            plt.xlabel(xlabel)
            plt.ylabel('Count')

    plt.title(f'{var_name} from {base_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    return plot_file

def generate_bar_chart(df, base_name, var_name, xlabel, output_dir):
    """
    genrates a, horizontal bar chart for top 10 values of numerical feature, indexes on y axis, values on x axis. we are not counting in this function, only listing largest values
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_file = out / f'{base_name}_{var_name}_bar_chart.png'

    data = df[var_name]


    if pd.api.types.is_numeric_dtype(data):
        top_values = data.nlargest(10)
        plt.figure(figsize=(10, 6))
        plt.barh(top_values.index.astype(str), top_values.values)
        plt.xlabel(xlabel)
        plt.ylabel('Feature')
        plt.savefig(plot_file)









    return plot_file


def load_train_summary(summary_path):
    """
    Load the latest entry from train_summary.json.
    """
    with open(summary_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    if not entries:
        raise ValueError(f"No entries in {summary_path}")
    return entries[0]



def build_time_to_review_df(predictions, advisories):

    rows = []
    # select only current year advisories from advisories
    advisories = {
        k: v for k, v in advisories.items()
        if v.get('published_at') and v['published_at'].year == today().year
    }



    for rec in predictions:
        ghsa = rec.get('ghsa_id')
        meta = advisories.get(ghsa)
        if not meta:
            continue

        # build only the keys we care about
        row = {
            'published_at': meta.get('published_at'),
            'ghsa_id': ghsa,
            'severity_label': meta.get('severity_label'),
            'cwe_ids': ", ".join(meta.get('cwe_ids', [])),
            'time_to_review': (
                round(rec.get('time_to_review'))
                if rec.get('time_to_review') is not None else None
            ),
            'time_to_nvd': meta.get('time_to_nvd'),
        }
        # add the hidden fields directly from meta
        # for k in hidden_fields:
        #     row[k] = meta.get(k)

        rows.append(row)

    df = pd.DataFrame(rows)
    df['published_at'] = pd.to_datetime(df['published_at']).dt.date
    df.sort_values('published_at', ascending=False, inplace=True)
    return df


def generate_section1_html(df, out_dir):
    """
    Gera a seção 1 do relatório, com a tabela vazia que será preenchida via AJAX
    pelo DataTables core e o histograma de time_to_review.
    """
    from pathlib import Path

    # garante que a pasta existe e gera o histograma
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_file = generate_histogram(
        df,
        base_name=f'inferred_data (year={today().year})',
        var_name='time_to_review',
        xlabel='until x days',
        output_dir=out_dir
    )

    # cabeçalho da tabela
    table_html = """
<table id="time_to_review_table" class="display nowrap" style="width:100%">
  <thead>
    <tr>
      <th>Published Date</th>
      <th>GHSA ID</th>
      <th>Severity</th>
      <th>CWE IDs</th>
      <th>Time to NVD</th>
      <th>Time to Review</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
"""

    return f"""
<details open>
  <summary><h2>Section 1: Time to Review Estimations for year {today().year} </h2></summary>

  {table_html}

  <h3>Distribution of Time to Review</h3>
  <img src="{plot_file.name}" alt="Distribution of Time to Review" style="max-width:100%;height:auto;" />

</details>
<!-- jQuery -->
<script src="{JQUERY_JS}"></script>
<!-- DataTables core JS -->
<script src="{DATA_TABLES_JS}"></script>
<!-- Scroller extension JS (para section1) -->
<script src="{SCROLLER_JS}"></script>
<script>
  // inicializa DataTable apenas se ainda não estiver inicializado
  $(function() {{
      $('#time_to_review_table').DataTable({{
        ajax: {{
            url: 'time_to_review_data.json',
            dataSrc: '',
            cache: false
        
        }},
        columns: [
          {{ data: 'published_at' }},
          {{ data: 'ghsa_id'        }},
          {{ data: 'severity_label' }},
          {{ data: 'cwe_ids'        }},
          {{ data: 'time_to_nvd'    }},
          {{ data: 'time_to_review' }}
        ],
        deferRender:  true,
        scrollY:      '60vh',
        scrollX:      true,
        scrollCollapse:true,
        scroller:     true,   // ativa virtual scrolling
        order:        [[0,'desc']]
      }});
  }});
</script>
"""






def generate_section2_html(out):
    df = load_normalized_advisories_df()


    total = len(df)

    # null percentage for whole base
    null_pct = (df.isnull().mean() * 100).round(2)
    zero_null_cols = null_pct[null_pct == 0]
    nonzero_null = null_pct[null_pct > 0]

    # --- Numerical summary (features como linhas) ---
    num_cols = ['time_to_review', 'time_to_nvd', 'num_cwes', 'num_refs', 'cvss_number']
    num_df = df[num_cols].describe().T.round(2)
    num_html = num_df.style.format("{:.2f}").set_table_attributes(
          'class="display nowrap" '
          'id="section2_num_summary" '
          'style="width:100%"'
      ).to_html()
    # --- Categorical severity ---
    cat_sev_df = df['severity_label'].value_counts().to_frame('count')
    cat_sev_html = cat_sev_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cat_severity" '
          'style="width:100%"'
      ).to_html()
    # --- CWE IDs (única tabela) ---
    cwe_df = df['cwe_ids'].explode().value_counts().to_frame('count')
    cwe_html = cwe_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cwe" '
          'style="width:100%"'
      ).to_html()
    # --- Ecosystem all catefories ---
    cat_ecos_df = df['ecosystem'].value_counts().to_frame('count')
    cat_ecos_html = cat_ecos_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cat_ecosystem" '
          'style="width:100%"'
      ).to_html()
    # --- Package names all categories ---
    cat_pkg_df = df['package_name'].value_counts().to_frame('count')
    cat_pkg_html = cat_pkg_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cat_package" '
          'style="width:100%"'
      ).to_html()
    # reviewed advisories subset
    dfr = df[df['time_to_review'].notnull()]
    total_r = len(dfr)

    # null percentage for reviewed advisories
    null_pct_r = (dfr.isnull().mean() * 100).round(2)
    zero_null_cols_r = null_pct_r[null_pct_r == 0]
    nonzero_null_r = null_pct_r[null_pct_r > 0]

    # reviewed advisories description tables
    num_cols_r = ['time_to_nvd', 'num_cwes', 'num_refs', 'cvss_number']
    num_df_r = dfr[num_cols_r].describe().T.round(2)
    num_html_r = num_df_r.style.format("{:.2f}").set_table_attributes(
          'class="display nowrap" '
          'id="section2_num_summary_r" '
          'style="width:100%"'
      ).to_html()
    cat_sev_r_df = dfr['severity_label'].value_counts().to_frame('count')
    cat_sev_r = cat_sev_r_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cat_severity_r" '
          'style="width:100%"'
      ).to_html()
    cat_ecos_r_df = dfr['ecosystem'].value_counts().to_frame('count')
    cat_ecos_r = cat_ecos_r_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cat_ecosystem_r" '
          'style="width:100%"'
      ).to_html()
    cat_pkg_r_df = dfr['package_name'].value_counts().head(10).to_frame('count')
    cat_pkg_r = cat_pkg_r_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cat_package_r" '
          'style="width:100%"'
      ).to_html()
    cwe_r_df = dfr['cwe_ids'].explode().value_counts().to_frame('count')
    cwe_r_html = cwe_r_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cwe_r" '
          'style="width:100%"'
      ).to_html()
    # unreviewed advisories subset
    dfu = df[df['time_to_review'].isnull()]
    total_u = len(dfu)

    # null percentage for unreviewed advisories
    null_pct_u = (dfu.isnull().mean() * 100).round(2)
    zero_null_cols_u = null_pct_u[null_pct_u == 0]
    nonzero_null_u = null_pct_u[null_pct_u > 0]

    # unreviewed advisories description tables
    num_cols_u = ['time_to_nvd', 'num_cwes', 'num_refs', 'cvss_number']
    num_df_u = dfu[num_cols_u].describe().T.round(2)
    num_html_u = num_df_u.style.format("{:.2f}").set_table_attributes(
          'class="display nowrap" '
          'id="section2_num_summary_u" '
          'style="width:100%"'
      ).to_html()
    cat_sev_u_df = dfu['severity_label'].value_counts().to_frame('count')
    cat_sev_u = cat_sev_u_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cat_severity_u" '
          'style="width:100%"'
      ).to_html()
    cwe_u_df = dfu['cwe_ids'].explode().value_counts().to_frame('count')
    cwe_u_html = cwe_u_df.style.set_table_attributes(
          'class="display nowrap" '
          'id="section2_cwe_u" '
          'style="width:100%"'
      ).to_html()

    # Gera histogramas
    hist_nvd = generate_histogram(df,'normalized_advisory', 'time_to_nvd', 'until x days', out)
    hist_time_to_review = generate_histogram(df, 'normalized_advisory', 'time_to_review', 'until x days', out)
    hist_num_cwes = generate_histogram(df, 'normalized_advisory', 'num_cwes', 'number of CWEs', out)
    hist_num_refs = generate_histogram(df, 'normalized_advisory', 'num_refs', 'number of references', out)
    hist_cvss_number = generate_histogram(df, 'normalized_advisory', 'cvss_number', 'CVSS score', out)
    hist_severity_label = generate_histogram(df, 'normalized_advisory', 'severity_label', 'severity label', out)
    hist_cwe_ids = generate_histogram(df, 'normalized_advisory', 'cwe_ids', 'CWE IDs', out)
    hist_ecosystem = generate_histogram(df, 'normalized_advisory', 'ecosystem', 'ecosystem', out)
    hist_pkg_name = generate_histogram(df, 'normalized_advisory', 'package_name', 'package name', out)

    hist_r_nvd = generate_histogram(dfr, 'normalized_advisory(reviewed)', 'time_to_nvd', 'until x days', out)
    hist_r_num_cwes = generate_histogram(dfr, 'normalized_advisory(reviewed)', 'num_cwes', 'number of CWEs', out)
    hist_r_num_refs = generate_histogram(dfr, 'normalized_advisory(reviewed)', 'num_refs', 'number of references', out)
    hist_r_cvss_number = generate_histogram(dfr, 'normalized_advisory(reviewed)', 'cvss_number', 'CVSS score', out)
    hist_r_severity_label = generate_histogram(dfr, 'normalized_advisory(reviewed)', 'severity_label', 'severity label', out)
    hist_r_cwe_ids = generate_histogram(dfr, 'normalized_advisory(reviewed)', 'cwe_ids', 'CWE IDs', out)
    hist_r_ecosystem = generate_histogram(dfr, 'normalized_advisory(reviewed)', 'ecosystem', 'ecosystem', out)
    hist_r_pkg_name = generate_histogram(dfr, 'normalized_advisory(reviewed)', 'package_name', 'package name', out)

    hist_u_nvd = generate_histogram(dfu, 'normalized_advisory(unreviewed)', 'time_to_nvd', 'until x days', out)
    hist_u_num_cwes = generate_histogram(dfu, 'normalized_advisory(unreviewed)', 'num_cwes', 'number of CWEs', out)
    hist_u_num_refs = generate_histogram(dfu, 'normalized_advisory(unreviewed)', 'num_refs', 'number of references', out)
    hist_u_severity_label = generate_histogram(dfu, 'normalized_advisory(unreviewed)', 'severity_label', 'severity label', out)
    hist_u_cwe_ids = generate_histogram(dfu, 'normalized_advisory(unreviewed)', 'cwe_ids', 'CWE IDs', out)


    # Wrappers que limitam largura a 80% da viewport
    num_wrapper       = f'<div style="max-width:80vw; margin-bottom:1em;">{num_html}</div>'
    cat_sev_wrapper   = f'<div style="max-width:40vw; margin-bottom:1em;">{cat_sev_html}</div>'
    cwe_wrapper       = f'<div style="max-width:40vw; margin-bottom:1em;">{cwe_html}</div>'
    cat_ecos_wrapper  = f'<div style="max-width:40vw; margin-bottom:1em;">{cat_ecos_html}</div>'
    cat_pkg_wrapper   = f'<div style="max-width:60vw; margin-bottom:1em;">{cat_pkg_html}</div>'

    num_r_wrapper     = f'<div style="max-width:80vw; margin-bottom:1em;">{num_html_r}</div>'
    cat_sev_r_wrapper = f'<div style="max-width:40vw; margin-bottom:1em;">{cat_sev_r}</div>'
    cat_ecos_r_wrap   = f'<div style="max-width:40vw; margin-bottom:1em;">{cat_ecos_r}</div>'
    cat_pkg_r_wrap    = f'<div style="max-width:60vw; margin-bottom:1em;">{cat_pkg_r}</div>'
    cwe_r_wrapper     = f'<div style="max-width:40vw; margin-bottom:1em;">{cwe_r_html}</div>'

    num_u_wrapper     = f'<div style="max-width:80vw; margin-bottom:1em;">{num_html_u}</div>'
    cat_sev_u_wrapper = f'<div style="max-width:40vw; margin-bottom:1em;">{cat_sev_u}</div>'
    cwe_u_wrapper     = f'<div style="max-width:40vw; margin-bottom:1em;">{cwe_u_html}</div>'




    section_html = f"""
<details>
    <summary><h2>Section 2: Normalized Advisories Base Stats</h2></summary>
    
    <h3>Null Percentage Summary</h3>
    <p><strong>Total records:</strong> {total}</p>
    
    <h3>Columns with 0% null</h3>
      <ul>
        {''.join(f'<li>{col}: {pct:.2f}%</li>' for col,pct in zero_null_cols.items())}
      </ul>
        
    <h3>Columns with null % > 0</h3>
      <ul>
        {''.join(f'<li>{col}: {pct:.2f}%</li>' for col,pct in nonzero_null.items())}
      </ul>
    
    
    <h3>Numerical Summary</h3>
    <details>
        <summary>Show/Hide Graphs</summary>
        
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_nvd.name}" alt="Time to NVD" style="width:50%;" />
        <img src="{hist_time_to_review.name}" alt="Time to Review" style="width:50%;" />
        </div>
        
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_num_cwes.name}" alt="Number of CWEs" style="width:50%;" />
        <img src="{hist_num_refs.name}" alt="Number of References" style="width:50%;" />
        </div>    
        
        <div style="display:flex; gap:1rem;margin-bottom:1em;">
        <img src="{hist_cvss_number.name}" alt="CVSS Number" style="width:50%;" />
        </div>
    </details>
    
    <details>
        <summary>Show/Hide tables</summary>
        {num_wrapper}
    </details>
    
    <h3>Categorical Summary – Severity Label</h3>
    
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
            <img src="{hist_severity_label.name}" alt="Severity Label" style="width:50%;" />
        </div>
    </details>
    
    <details>
        <summary>Show/Hide Tables</summary>
        {cat_sev_wrapper}
    </details>
    
    <h3>Categorical Summary – CWE IDs</h3>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_cwe_ids.name}" alt="CWE IDs" style="width:50%;" />
        </div>
    </details>
    
    <details>
        <summary>Show/Hide Tables</summary>
        {cwe_wrapper}
    </details>
    
    <h3>Categorical Summary – Ecosystem</h3>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_ecosystem.name}" alt="Ecosystem" style="width:50%;" />
        </div>
    </details>
    <details>
        <summary>Show/Hide Tables</summary>
        {cat_ecos_wrapper}
    </details>
    
    <h3>Categorical Summary – Package Name</h3>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_pkg_name.name}" alt="Ecosystem" style="width:50%;" />
        </div>
    </details>
    <details>
        <summary>Show/Hide Tables</summary>
        {cat_pkg_wrapper}
    </details>
        
    <hr/>
    
    <h3> Reviewed Advisories</h3>
    
    <h3>Null Percentage Summary</h3>
    <p><strong>Total records:</strong> {total_r}</p>
    <h3>Columns with 0% null</h3>
      <ul>
        {''.join(f'<li>{col}: {pct:.2f}%</li>' for col,pct in zero_null_cols_r.items())}
      </ul>
    <h3>Columns with null % > 0</h3>
    <ul>
        {''.join(f'<li>{col}: {pct:.2f}%</li>' for col,pct in nonzero_null_r.items())}
    </ul>
    
    
        
    
    <h4>Numerical Summary</h4>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_r_nvd.name}" alt="Time to NVD (reviewed)" style="width:50%;" />
        <img src="{hist_r_cvss_number.name}" alt="CVSS Number (reviewed)" style="width:50%;" />
        </div>
        
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_r_num_cwes.name}" alt="Number of CWEs (reviewed)" style="width:50%;" />
        <img src="{hist_r_num_refs.name}" alt="Number of References (reviewed)" style="width:50%;" />
        </div>
    </details>
    <details>
        <summary>Show/Hide Tables</summary>
        {num_r_wrapper}
    </details>

    
    <h4>Categorical Summary – Severity Label</h4>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_r_severity_label.name}" alt="Severity Label (reviewed)" style="width:50%;" />
        </div>
    </details>
    
    <details>   
        <summary>Show/Hide Tables</summary>
        {cat_sev_r_wrapper}
    </details>
        
    <h4>Categorical Summary – CWE IDs</h4>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
            <img src="{hist_r_cwe_ids.name}" alt="CWE IDs (reviewed)" style="width:50%;" />
        </div>        
    </details>
    <details>
        <summary>Show/Hide Tables</summary>
        {cwe_r_wrapper}
    </details>
    
    <h4>Categorical Summary – Ecosystem</h4>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
            <img src="{hist_r_ecosystem.name}" alt="Ecosystem (reviewed)" style="width:50%;" />
        </div>
    </details>
    <details>
        <summary>Show/Hide Tables</summary>
        {cat_ecos_r_wrap}
    </details>
    
    <h4>Categorical Summary – Package Name</h4>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
            <img src="{hist_r_pkg_name.name}" alt="Package Name (reviewed)" style="width:50%;" />
        </div>
    </details>
    <details>
        <summary>Show/Hide Tables</summary>
        {cat_pkg_r_wrap}
    </details>
    
    <hr/>
    <h3>Unreviewed Advisories</h3>
    <h3>Null Percentage Summary</h3>
    <p><strong>Total records:</strong> {total_u}</p>
    <h3>Columns with 0% null</h3>
      <ul>
        {''.join(f'<li>{col}: {pct:.2f}%</li>' for col,pct in zero_null_cols_u.items())}
      </ul>
    <h3>Columns with null % > 0</h3>
    <ul>
        {''.join(f'<li>{col}: {pct:.2f}%</li>' for col,pct in nonzero_null_u.items())}
    </ul>
    
    <h4>Numerical Summary</h4>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_u_nvd.name}" alt="Time to NVD (unreviewed)" style="width:50%;" />
        </div>
        
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_u_num_cwes.name}" alt="Number of CWEs (unreviewed)" style="width:50%;" />
        <img src="{hist_u_num_refs.name}" alt="Number of References (unreviewed)" style="width:50%;" />
        </div>
    </details>
    
    <details>
        <summary>Show/Hide Tables</summary>
        {num_u_wrapper}
    </details>
    <h4>Categorical Summary – Severity Label</h4>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_u_severity_label.name}" alt="Severity Label (unreviewed)" style="width:50%;" />
        </div>
    </details>
    <details>
        <summary>Show/Hide Tables</summary>
        {cat_sev_u_wrapper}
    </details>
    
    <h4>Categorical Summary – CWE IDs</h4>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_u_cwe_ids.name}" alt="CWE IDs (unreviewed)" style="width:50%;" />
        </div>
    </details>
    
    <details>
        <summary>Show/Hide Tables</summary>
        {cwe_u_wrapper}
        
    </details>
    
</details>

<script>
  $(document).ready(function() {{
    [ 
      'section2_num_summary',
      'section2_cat_severity',
      'section2_cwe',
        'section2_cat_ecosystem',
        'section2_cat_package',
      'section2_num_summary_r',
      'section2_cat_severity_r',
      'section2_cat_ecosystem_r',
      'section2_cat_package_r',
      'section2_cwe_r',
        'section2_num_summary_u',
        'section2_cat_severity_u',
        'section2_cwe_u'
    ].forEach(function(id) {{
      $('#'+id).DataTable({{
        scrollY:        '40vh',      // altura do painel
        scrollX:        true,        // rolagem horiz.
        scrollCollapse: true,
        paging:         false,       // sem paginação
      }});
    }});
  }});
</script>
"""
    return section_html





def generate_section3_html(out):
    """
    Section 3: Training Info
    """
    # load data...
    df_normalized = load_normalized_advisories_df()
    df_normalized_reviewd = df_normalized.dropna(subset=['time_to_review'])
    df_normalized_unreviewed = df_normalized.loc[df_normalized['time_to_review'].isnull()]
    df_train = pd.read_parquet(BASE_DIR / 'data/processed/features_time_to_review_train.parquet')
    df_infer = pd.read_parquet(BASE_DIR / 'data/processed/features_time_to_review_infer.parquet')
    train_sum = load_train_summary(BASE_DIR / 'data/processed/time_to_review_model/train_summary.json')

    nunique = df_train.nunique()
    binary_feats = nunique[nunique <= 2].index.tolist()
    nonbinary_feats = nunique[nunique > 2].index.tolist()

    # — Train non-binary
    train_nonbin_html = (
        df_train[nonbinary_feats]
        .describe().T.round(3)
        .style
        .format("{:.2f}").format("{:.2f}").set_table_attributes(
          'class="display nowrap" '
          'id="section3_train_nonbin" '
          'style="width:100%"'
      ).to_html()
    )

    # — Train binary
    df_train_bin = df_train[binary_feats].agg(['mean', 'std']).T.round(3)
    train_bin_html = (
        df_train_bin
        .style
        .format("{:.2f}").set_table_attributes(
          'class="display nowrap" '
          'id="section3_train_bin" '
          'style="width:100%"'
      ).to_html()
    )

    # — Correlation

    df_corr = df_train.corr()['time_to_review'].drop('time_to_review').abs().sort_values(ascending=False).to_frame('corr_with_target').round(3)
    corr_html = (
        df_corr
        .style
        .format("{:.2f}").set_table_attributes(
          'class="display nowrap" '
          'id="section3_corr" '
          'style="width:100%"'
      ).to_html()
    )

    # — RF importances
    X, y = df_train.drop(columns=['time_to_review'], errors='ignore'), df_train['time_to_review']
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    df_fi = pd.DataFrame({'importance': rf.feature_importances_}, index=X.columns).sort_values('importance', ascending=False).round(3)
    fi_html = (
        df_fi
        .style
        .format("{:.2f}").set_table_attributes(
          'class="display nowrap" '
          'id="section3_fi" '
          'style="width:100%"'
      ).to_html()
    )

    # — Infer non-binary & binary
    infer_nonbin_html = (
        df_infer[nonbinary_feats]
        .describe().T.round(3)
        .style.format("{:.2f}").set_table_attributes(
          'class="display nowrap" '
          'id="section3_infer_nonbin" '
          'style="width:100%"'
      ).to_html()
    )

    df_infer_bin = df_infer[binary_feats].agg(['mean', 'std']).T.round(3)
    infer_bin_html = (
        df_infer[binary_feats]
        .agg(['mean', 'std']).T.round(3)
        .style
        .format("{:.2f}").set_table_attributes(
          'class="display nowrap" '
          'id="section3_infer_bin" '
          'style="width:100%"'
      ).to_html()
    )

    # Model performance
    skip = {'timestamp', 'bestmodel', 'capcutceil', 'quantiles'}
    models = {k: v for k, v in train_sum.items() if k not in skip and isinstance(v, dict)}
    metrics = {
        name: {'MAE': val['MAE'], 'RMSE': val['RMSE'], 'R2': val['R2'], 'Fit Time': val['mean_fit_time'], 'Score Time': val['mean_score_time'], 'CV Folds': val['cv_folds']}
        for name, val in models.items()
    }
    metrics_html = (
        pd.DataFrame.from_dict(metrics, orient='index')
        .round(3)
        .style
        .format("{:.3f}")
        .set_table_attributes(
            'class="display nowrap" '
            'id="section3_metrics" '
            'style="width:100%"'
        ).to_html()
    )


    # wrappers that limit width of viewport

    metrics_wrapper = f'<div style="max-width:40vw; margin-bottom:1em;">{metrics_html}</div>'
    train_nonbin_wrapper = f'<div style="max-width:80vw; margin-bottom:1em;">{train_nonbin_html}</div>'
    train_bin_wrapper = f'<div style="max-width:40vw; margin-bottom:1em;">{train_bin_html}</div>'
    corr_wrapper = f'<div style="max-width:30vw; margin-bottom:1em;">{corr_html}</div>'
    fi_wrapper = f'<div style="max-width:30vw; margin-bottom:1em;">{fi_html}</div>'
    infer_nobin_wrapper = f'<div style="max-width:80vw; margin-bottom:1em;">{infer_nonbin_html}</div>'
    infer_bin_wrapper = f'<div style="max-width:30vw; margin-bottom:1em;">{infer_bin_html}</div>'




    best = train_sum.get('bestmodel')
    cap = train_sum.get('capcutceil')
    quants = train_sum.get('quantiles')
    fit_params = json.dumps(train_sum.get(best).get('best_params', {}))

    # --- extrai dia da semana, dia do mês e dia do ano ---

    df_train['day_of_week'] = df_normalized_reviewd['published_at'].dt.weekday + 1
    df_train['day_of_month'] = df_normalized_reviewd['published_at'].dt.day
    df_train['day_of_year'] = df_normalized_reviewd['published_at'].dt.dayofyear

    # --- ge
    # gera histogramas para dados de treino---
    hist_dow_train = generate_histogram(df_train, 'train_features', 'day_of_week', 'Day of Week', out)
    hist_dom_train = generate_histogram(df_train, 'train_features', 'day_of_month', 'Day of Month', out)
    hist_doy_train = generate_histogram(df_train, 'train_features', 'day_of_year', 'Day of Year', out)

    hist_details_train = generate_histogram(
        df_train,
        base_name='normalized_advisory',
        var_name='details_char_count',
        xlabel='Details Char Count',
        output_dir=out
    )

    # bar charts to rank top 10 correlated features, fetures with highest importance, and mean values of train binary features
    bar_train_corr = generate_bar_chart(df_corr, 'train_features', 'corr_with_target', 'Feature Correlation with time_to_review', out)
    bar_train_fi = generate_bar_chart(df_fi, 'train_features', 'importance', 'Feature Importance', out)
    bar_train_bin = generate_bar_chart(df_train_bin, 'train_features', 'mean', 'Mean of Binary Features', out)




    # gera histogramas para dados de inferência
    df_infer['day_of_week'] = df_normalized_unreviewed['published_at'].dt.weekday + 1
    df_infer['day_of_month'] = df_normalized_unreviewed['published_at'].dt.day
    df_infer['day_of_year'] = df_normalized_unreviewed['published_at'].dt.dayofyear

    hist_dow_infer = generate_histogram(df_infer, 'infer_features', 'day_of_week', 'Day of Week', out)
    hist_dom_infer = generate_histogram(df_infer, 'infer_features', 'day_of_month', 'Day of Month', out)
    hist_doy_infer = generate_histogram(df_infer, 'infer_features', 'day_of_year', 'Day of Year', out)

    hist_details_infer = generate_histogram(
        df_infer,
        base_name='normalized_advisory',
        var_name='details_char_count',
        xlabel='Details Char Count',
        output_dir=out
    )

    # bar charts to rank top 10 mean values of infer binary features
    bar_infer_bin = generate_bar_chart(df_infer_bin, 'infer_features', 'mean', 'Mean of Binary Features', out)




    # assemble section
    section3 = f"""
<details>
  <summary><h2>Section 3: Training Info</h2></summary>
  
  <h3>Model Performance Summary</h3>
  {metrics_wrapper}
  <p><strong>Best model:</strong> {best}</p>
  <p><strong>Cap/Ceil:</strong> {cap}</p>
  <p><strong>Cap Quantiles:</strong> {quants}</p>
    <p><strong>Fit Params:</strong> {fit_params}</p>

  <h3>Train Data: Non-binary Features</h3>
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_dow_train.name}" alt="Day of Week" style="width:50%;" />
        <img src="{hist_dom_train.name}" alt="Day of Month" style="width:50%;" />
        </div>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_doy_train.name}" alt="Day of Year" style="width:50%;" />
        <img src="{hist_details_train.name}" alt="Details Char Count" style="width:50%;" />
        </div>
    </details>
  <details>
    <summary>Show/Hide Table</summary>
  {train_nonbin_wrapper}
  </details>

  <h3>Train Data: Binary Features</h3>
  <details>
    <summary>Show/Hide Graphs</summary>
    <div style="display:flex; gap:1rem; margin-bottom:1em;">
    <img src="{bar_train_bin.name}" alt="Mean of Binary Features" style="width:50%;" />
    </div>
    </details>
  
  <details>
    <summary>Show/Hide Table</summary>
  {train_bin_wrapper}
    </details>

  <h3>Feature Correlation with time_to_review</h3>
  <details>
    <summary>Show/Hide Graphs</summary>
    <div style="display:flex; gap:1rem; margin-bottom:1em;">
    <img src="{bar_train_corr.name}" alt="Feature Correlation" style="width:50%;" />
    </div>
    </details>
  
  <details>
    <summary>Show/Hide Table</summary>
  {corr_wrapper}
    </details>

  <h3>RandomForest Feature Importances</h3>
  <details>
    <summary>Show/Hide Graphs</summary>
    <div style="display:flex; gap:1rem; margin-bottom:1em;">
    <img src="{bar_train_fi.name}" alt="Feature Importance" style="width:50%;" />
    </div>
  </details>
  
  <details>
    <summary>Show/Hide Table</summary>
  {fi_wrapper}
  </details>

  <h3>Infer Data: Non-binary Features</h3>
  
    <details>
        <summary>Show/Hide Graphs</summary>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_dow_infer.name}" alt="Day of Week" style="width:50%;" />
        <img src="{hist_dom_infer.name}" alt="Day of Month" style="width:50%;" />
        </div>
        <div style="display:flex; gap:1rem; margin-bottom:1em;">
        <img src="{hist_doy_infer.name}" alt="Day of Year" style="width:50%;" />
        <img src="{hist_details_infer.name}" alt="Details Char Count" style="width:50%;" />
        </div>  
    </details>
  <details>
    <summary>Show/Hide Table</summary>
  {infer_nobin_wrapper}
    </details>

  <h3>Infer Data: Binary Features</h3>
  <details>
    <summary>Show/Hide Graphs</summary>
    <div style="display:flex; gap:1rem; margin-bottom:1em;">
    <img src="{bar_infer_bin.name}" alt="Mean of Binary Features" style="width:50%;" />
    </div>
    </details>
  
  <details>
    <summary>Show/Hide Table</summary>
  {infer_bin_wrapper}
  </details>
</details>

<script>
  $(document).ready(function() {{
    [ 
      'section3_train_nonbin',
      'section3_train_bin',
      'section3_corr',
        'section3_fi',
        'section3_infer_nonbin',
      'section3_infer_bin',
      'section3_metrics',
      
    ].forEach(function(id) {{
      $('#'+id).DataTable({{
        scrollY:        '40vh',      // altura do painel
        scrollX:        true,        // rolagem horiz.
        scrollCollapse: true,
        paging:         false,       // sem paginação
      }});
    }});
  }});
</script>
"""
    return section3


def run(output_dir=None):
    data_dir = BASE_DIR / 'data' / 'processed' / 'time_to_review_model'
    rpt_dir  = Path(config.get('reports_path', BASE_DIR / 'reports'))
    out      = Path(output_dir) if output_dir else rpt_dir / 'time_to_review'
    out.mkdir(parents=True, exist_ok=True)

    # Section 1 data
    ts      = get_latest_model_timestamp(data_dir / 'train_summary.json')
    preds   = load_predictions(ts)
    df_adv  = load_normalized_advisories_df()
    advs    = {r['ghsa_id']: r for r in df_adv.to_dict(orient='records')}
    df1     = build_time_to_review_df(preds, advs)

    # dump the JSON array
    json_path = out / 'time_to_review_data.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            df1.to_dict(orient='records'),
            f,
            default=str   # auto-convert Timestamps/dates to strings
        )

    # now generate sections (note: section1 no longer needs records_js)
    section1 = generate_section1_html(df1, out)
    logger.info("Section 1 generated")
    section2 = generate_section2_html(out)
    logger.info("Section 2 generated")
    section3 = generate_section3_html(out)
    logger.info("Section 3 generated")

    # assemble HTML (make sure jQuery/DataTables loaded before inline scripts)
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <!-- DataTables CSS -->
  <link rel="stylesheet" href="{DATA_TABLES_CSS}" />
  <!-- Scroller CSS (para section1) -->
  <link rel="stylesheet" href="{SCROLLER_CSS}" />
</head>
<body>
  
  {section1}
  {section2}
  {section3}

<script>
  // Depois de inicializar todas as suas DataTables…
  document.querySelectorAll('details').forEach(function(d){{
    d.addEventListener('toggle', function(){{
      // para cada <table class="display"> dentro do <details> recém-aberto…
      d.querySelectorAll('table.display').forEach(function(tbl){{
        // chama columns.adjust()
        $(tbl).DataTable().columns.adjust();
      }});
    }});
  }});
</script> 
  
</body>


</html>
"""
    report_file = out / 'time_to_review_report.html'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Report saved to {report_file}")
    return html




if __name__ == '__main__':
    run()