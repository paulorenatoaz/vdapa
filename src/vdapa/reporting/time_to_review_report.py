import math
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from vdapa.config import config, BASE_DIR
from vdapa.utils import setup_logging

logger = setup_logging("reporting", "time_to_review_report")

# CDN Links
DATA_TABLES_CSS = "https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css"
SCROLLER_CSS = "https://cdn.datatables.net/scroller/2.2.1/css/scroller.dataTables.min.css"
JQUERY_JS = "https://code.jquery.com/jquery-3.6.0.min.js"
DATA_TABLES_JS = "https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"
SCROLLER_JS = "https://cdn.datatables.net/scroller/2.2.1/js/dataTables.scroller.min.js"


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




def generate_histogram_time_to_review_preds(df, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_file = out / 'time_to_review_histogram_predictions.png'
    plt.figure()
    plt.hist(df['time_to_review'].dropna(), bins=30)
    plt.xlabel('Time to Review (until x days)')
    plt.ylabel('Count')
    plt.title('Histogram of Time to Review')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    return plot_file

def generate_histogram_time_to_review_normalized(df, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_file = out / 'time_to_review_histogram_normalized.png'
    plt.figure()
    plt.hist(df['time_to_review'].dropna(), bins=30)
    plt.xlabel('Time to Review (until x days)')
    plt.ylabel('Count')
    plt.title('Histogram of Time to Review (Normalized)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    return plot_file

def generate_histograms_time_to_nvd(df, dfr, output_dir):
    out = Path(output_dir)

    # Histogram for all records
    out.mkdir(parents=True, exist_ok=True)
    plot_file = out / 'time_to_nvd_histogram.png'
    plt.figure()
    plt.hist(df['time_to_nvd'].dropna(), bins=50)
    plt.xlabel('Time to NVD (until x days)')
    plt.ylabel('Count')
    plt.title('Histogram of Time to NVD')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    # Histogram for records with time_to_review
    plot_file_r = out / 'time_to_nvd_histogram_with_time_to_review.png'
    plt.figure()
    plt.hist(dfr['time_to_nvd'].dropna(), bins=50)
    plt.xlabel('Time to NVD (until x days)')
    plt.ylabel('Count')
    plt.title('Histogram of Time to NVD (with time_to_review)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file_r)
    plt.close()



    return plot_file, plot_file_r

def load_train_summary(summary_path):
    """
    Load the latest entry from train_summary.json.
    """
    with open(summary_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    if not entries:
        raise ValueError(f"No entries in {summary_path}")
    return entries[0]


# def build_time_to_review_df(predictions, advisories):
#     rows = []
#     for rec in predictions:
#         ghsa = rec.get('ghsa_id')
#         meta = advisories.get(ghsa)
#         if not meta:
#             continue
#         rows.append({
#             'published_date': meta.get('published_at'),
#             'ghsa_id': ghsa,
#             'severity_label': meta.get('severity_label'),
#             'cwe_ids': ", ".join(meta.get('cwe_ids', [])),
#             'time_to_review': round(rec.get('time_to_review')) if rec.get('time_to_review') is not None else None
#         })
#     df = pd.DataFrame(rows)
#     df['published_date'] = pd.to_datetime(df['published_date']).dt.date
#     df.sort_values('published_date', ascending=False, inplace=True)
#     return df
#
#
# def generate_section1_html(out, df, records_js):
#     headers = ['Published Date', 'GHSA ID', 'Severity', 'CWE IDs', 'Time to Review']
#     thead = '<thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead>'
#     table_html = f'<table id="time_to_review_table" class="display" style="width:100%">{thead}<tbody></tbody></table>'
#     plot_file = generate_histogram_time_to_review_preds(df, out)
#     section_html = f"""
# <details open>
#   <summary><h2>Section 1: Time to Review Estimations</h2></summary>
#   {table_html}
#   <h3>Distribution of Time to Review</h3>
#   <img src="{plot_file.name}" alt="Distribution of Time to Review" />
# </details>
# <script>
#   var timeData = {records_js};
#   $(document).ready(function() {{
#     $('#time_to_review_table').DataTable({{
#       data: timeData,
#       columns: [
#         {{ data: 'published_date' }},
#         {{ data: 'ghsa_id' }},
#         {{ data: 'severity_label' }},
#         {{ data: 'cwe_ids' }},
#         {{ data: 'time_to_review' }}
#       ],
#       deferRender: true,
#       scrollY: '60vh',
#       scroller: true,
#       pageLength: 25,
#       order: [[0,'desc']]
#     }});
#   }});
# </script>
# """
#     return section_html
def build_time_to_review_df(predictions, advisories):
    # We'll show these six fields...
    show_fields = [
        'published_date',
        'ghsa_id',
        'severity_label',
        'cwe_ids',
        'time_to_review',
        'time_to_nvd',
    ]
    # …and keep these four hidden for the responsive modal
    hidden_fields = [
        'cve_ids',
        'details',
        'references',
        'nvd_published_at',
    ]
    rows = []
    # select only 2025 advisories from advisories
    advisories = {
        k: v for k, v in advisories.items()
        if v.get('published_at') and v['published_at'].year == 2025
    }



    for rec in predictions:
        ghsa = rec.get('ghsa_id')
        meta = advisories.get(ghsa)
        if not meta:
            continue

        # build only the keys we care about
        row = {
            'published_date': meta.get('published_at'),
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
        for k in hidden_fields:
            row[k] = meta.get(k)

        rows.append(row)

    df = pd.DataFrame(rows)
    df['published_date'] = pd.to_datetime(df['published_date']).dt.date
    df.sort_values('published_date', ascending=False, inplace=True)
    return df




def generate_section1_html(df1, out):
    # generate the histogram as before
    plot_file = generate_histogram_time_to_review_preds(df1, out)

    # simple table at 90% width, centered, with six explicit headers
    table_html = (
        '<table id="time_to_review_table" class="display nowrap" '
        'style="width:90%; margin:auto;">'
        '<thead><tr>'
          '<th>Published Date</th>'
          '<th>GHSA ID</th>'
          '<th>Severity</th>'
          '<th>CWE IDs</th>'
          '<th>Time to NVD</th>'
          '<th>Time to Review</th>'
        '</tr></thead>'
        '<tbody></tbody>'
        '</table>'
    )

    return f"""
<details open>
  <summary><h2>Section 1: Time to Review Estimations</h2></summary>
  {table_html}
  <h3>Distribution of Time to Review</h3>
  <img src="{plot_file.name}" alt="Distribution of Time to Review" />
</details>

<script>
  $(document).ready(function() {{
    $('#time_to_review_table').DataTable({{
      ajax: {{
        url: 'time_to_review_data.json',
        dataSrc: ''
      }},
      columns: [
        {{ data: 'published_date',    title: 'Published Date' }},
        {{ data: 'ghsa_id',           title: 'GHSA ID' }},
        {{ data: 'severity_label',    title: 'Severity' }},
        {{ data: 'cwe_ids',           title: 'CWE IDs' }},
        {{ data: 'time_to_nvd',       title: 'Time to NVD' }},
        {{ data: 'time_to_review',    title: 'Time to Review' }}
      ],
      deferRender:    true,
      scrollY:        '60vh',
      scrollX:        true,
      scrollCollapse: true,
      autoWidth:      false,   // size to contents, don’t stretch
      pageLength:     25,
      order: [[0,'desc']]
    }});
  }});
</script>
"""

def generate_section2_html(out):
    df = load_normalized_advisories_df()

    total = len(df)
    null_pct = (df.isnull().mean() * 100).round(2)
    zero_null_cols = null_pct[null_pct == 0].index.tolist()

    # --- Numerical summary (features como linhas) ---
    num_cols = ['time_to_review', 'time_to_nvd', 'num_cwes', 'num_refs']
    num_df = df[num_cols].describe().T.round(2)
    num_html = num_df.to_html(
        classes='display',
        table_id='section2_num_summary',
    )

    # --- Categorical severity ---
    cat_sev_df = df['severity_label'].value_counts().to_frame('count')
    cat_sev_html = cat_sev_df.to_html(
        classes='display',
        table_id='section2_cat_severity',
    )

    # --- CWE IDs (única tabela) ---
    cwe_df = df['cwe_ids'].explode().value_counts().to_frame('count')
    cwe_html = cwe_df.to_html(
        classes='display',
        table_id='section2_cwe',
    )

    # --- Subconjunto com time_to_review ---
    dfr = df[df['time_to_review'].notnull()]
    total_r = len(dfr)

    num_cols_r = ['time_to_nvd', 'num_cwes', 'num_refs', 'cvss_number']
    num_df_r = dfr[num_cols_r].describe().T.round(2)
    num_html_r = num_df_r.to_html(
        classes='display',
        table_id='section2_num_summary_r',
    )

    cat_sev_r_df = dfr['severity_label'].value_counts().to_frame('count')
    cat_sev_r = cat_sev_r_df.to_html(
        classes='display',
        table_id='section2_cat_severity_r',
    )

    cat_ecos_r_df = dfr['ecosystem'].value_counts().to_frame('count')
    cat_ecos_r = cat_ecos_r_df.to_html(
        classes='display',
        table_id='section2_cat_ecosystem_r',
    )

    cat_pkg_r_df = dfr['package_name'].value_counts().head(10).to_frame('count')
    cat_pkg_r = cat_pkg_r_df.to_html(
        classes='display',
        table_id='section2_cat_package_r',
    )

    cwe_r_df = dfr['cwe_ids'].explode().value_counts().to_frame('count')
    cwe_r_html = cwe_r_df.to_html(
        classes='display',
        table_id='section2_cwe_r',
    )

    # Gera histogramas
    hist_nvd, hist_nvd_r = generate_histograms_time_to_nvd(df, dfr, out)
    hist_norm = generate_histogram_time_to_review_normalized(df, out)

    # Wrappers que limitam largura a 80% da viewport
    num_wrapper       = f'<div style="max-width:80vw; margin-bottom:1em;">{num_html}</div>'
    cat_sev_wrapper   = f'<div style="max-width:80vw; margin-bottom:1em;">{cat_sev_html}</div>'
    cwe_wrapper       = f'<div style="max-width:80vw; margin-bottom:1em;">{cwe_html}</div>'
    num_r_wrapper     = f'<div style="max-width:80vw; margin-bottom:1em;">{num_html_r}</div>'
    cat_sev_r_wrapper = f'<div style="max-width:80vw; margin-bottom:1em;">{cat_sev_r}</div>'
    cat_ecos_r_wrap   = f'<div style="max-width:80vw; margin-bottom:1em;">{cat_ecos_r}</div>'
    cat_pkg_r_wrap    = f'<div style="max-width:80vw; margin-bottom:1em;">{cat_pkg_r}</div>'
    cwe_r_wrapper     = f'<div style="max-width:80vw; margin-bottom:1em;">{cwe_r_html}</div>'

    section_html = f"""
<details>
  <summary><h2>Section 2: Normalized Advisories Base Stats</h2></summary>

  <div style="display:flex; gap:1rem; margin-bottom:1em;">
    <img src="{hist_nvd.name}" alt="Time to NVD" style="width:50%;" />
    <img src="{hist_norm.name}" alt="Time to Review" style="width:50%;" />
  </div>

  <p><strong>Total records:</strong> {total}</p>
  <p><strong>Columns with 0% null:</strong> {', '.join(zero_null_cols)}</p>

  <h3>Numerical Summary</h3>
  {num_wrapper}

  <h3>Categorical Summary – Severity Label</h3>
  {cat_sev_wrapper}

  <h3>Categorical Summary – CWE IDs</h3>
  {cwe_wrapper}

  <hr/>

  <h3>Records with <em>time_to_review</em> (Total: {total_r})</h3>

  <h4>Numerical Summary</h4>
  {num_r_wrapper}

  <h4>Categorical Summary – Severity Label</h4>
  {cat_sev_r_wrapper}

  <h4>Categorical Summary – Ecosystem</h4>
  {cat_ecos_r_wrap}

  <h4>Categorical Summary – Package Name</h4>
  {cat_pkg_r_wrap}

  <h4>Categorical Summary – CWE IDs</h4>
  {cwe_r_wrapper}

</details>

<script>
  $(document).ready(function() {{
    [ 
      'section2_num_summary',
      'section2_cat_severity',
      'section2_cwe',
      'section2_num_summary_r',
      'section2_cat_severity_r',
      'section2_cat_ecosystem_r',
      'section2_cat_package_r',
      'section2_cwe_r'
    ].forEach(function(id) {{
      $('#'+id).DataTable({{
        scrollY: '40vh',
        scrollCollapse: true,
        paging: false,
        autoWidth: false,
        scrollX: true
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
        .to_html(classes='display', table_id='section3_train_nonbin')
    )

    # — Train binary
    train_bin_html = (
        df_train[binary_feats]
        .agg(['mean', 'std']).T.round(3)
        .to_html(classes='display', table_id='section3_train_bin')
    )

    # — Correlation
    corr_html = (
        df_train.corr()['time_to_review']
        .drop('time_to_review').abs().sort_values(ascending=False)
        .to_frame('corr_with_target').round(3)
        .to_html(classes='display', table_id='section3_corr')
    )

    # — RF importances
    X, y = df_train.drop(columns=['time_to_review'], errors='ignore'), df_train['time_to_review']
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    fi_html = (
        pd.DataFrame({'importance': rf.feature_importances_}, index=X.columns)
        .sort_values('importance', ascending=False).round(3)
        .to_html(classes='display', table_id='section3_fi')
    )

    # — Infer non-binary & binary
    infer_nonbin_html = (
        df_infer[nonbinary_feats]
        .describe().T.round(3)
        .to_html(classes='display', table_id='section3_infer_nonbin')
    )
    infer_bin_html = (
        df_infer[binary_feats]
        .agg(['mean', 'std']).T.round(3)
        .to_html(classes='display', table_id='section3_infer_bin')
    )

    # assemble section
    section3 = f"""
<details>
  <summary><h2>Section 3: Training Info</h2></summary>

  <h3>Train Data: Non-binary Features</h3>
  {train_nonbin_html}

  <h3>Train Data: Binary Features</h3>
  {train_bin_html}

  <h3>Feature Correlation with time_to_review</h3>
  {corr_html}

  <h3>RandomForest Feature Importances</h3>
  {fi_html}

  <h3>Infer Data: Non-binary Features</h3>
  {infer_nonbin_html}

  <h3>Infer Data: Binary Features</h3>
  {infer_bin_html}
</details>

<script>
  $(document).ready(function() {{
    $('#section3_train_nonbin').DataTable({{ scrollY: '30vh', paging: false }});
    $('#section3_train_bin').DataTable();
    $('#section3_corr').DataTable({{ paging: false }});
    $('#section3_fi').DataTable({{ scrollY: '30vh', pageLength: 10 }});
    $('#section3_infer_nonbin').DataTable({{ scrollY: '30vh', paging: false }});
    $('#section3_infer_bin').DataTable();
  }});
</script>
"""
    return section3


# def run_section(output_dir=None):
#     data_dir = BASE_DIR / 'data' / 'processed' / 'time_to_review_model'
#     rpt_dir = Path(config.get('reports_path', BASE_DIR / 'reports'))
#     out = Path(output_dir) if output_dir else rpt_dir / 'time_to_review'
#     out.mkdir(parents=True, exist_ok=True)
#
#     # Section 1
#     ts = get_latest_model_timestamp(data_dir / 'train_summary.json')
#     preds = load_predictions(ts)
#     df_advs = load_normalized_advisories_df()
#     advs = {rec.get('ghsa_id'): rec for rec in df_advs.to_dict(orient='records')}
#     df1 = build_time_to_review_df(preds, advs)
#     df1_copy = df1.copy()
#     df1_copy['published_date'] = df1_copy['published_date'].astype(str)
#     records_js = json.dumps(df1_copy.to_dict(orient='records'))
#     section1 = generate_section1_html(out, df1, records_js)
#
#     # Section 2
#     section2 = generate_section2_html(out)
#
#     # Section 3
#     section3 = generate_section3_html(out)
#
#     # Compose full HTML
#     html = f"""
# <!DOCTYPE html>
# <html>
# <head>
#   <meta charset="utf-8" />
#   <link rel="stylesheet" href="{DATA_TABLES_CSS}" />
#   <link rel="stylesheet" href="{SCROLLER_CSS}" />
#   <script src="{JQUERY_JS}"></script>
#   <script src="{DATA_TABLES_JS}"></script>
#   <script src="{SCROLLER_JS}"></script>
# </head>
# <body>
# {section1}
# {section2}
# {section3}
# </body>
# </html>
# """
#
#     report_file = out / 'time_to_review_report.html'
#     with open(report_file, 'w', encoding='utf-8') as f:
#         f.write(html)
#     logger.info(f"Report saved to {report_file}")
#     return html
def run_section(output_dir=None):
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
    section2 = generate_section2_html(out)
    section3 = generate_section3_html(out)

    # assemble HTML (make sure jQuery/DataTables loaded before inline scripts)
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="{DATA_TABLES_CSS}" />
</head>
<body>
  <script src="{JQUERY_JS}"></script>
  <script src="{DATA_TABLES_JS}"></script>

  {section1}
  {section2}
  {section3}
</body>
</html>
"""
    report_file = out / 'time_to_review_report.html'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Report saved to {report_file}")
    return html



def main():
    run_section()


if __name__ == '__main__':
    main()