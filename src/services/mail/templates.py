from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class EmailPayload:
    to: str
    subject: str
    body_html: str
    body_text: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d · %H:%M UTC")


def _build_html(
    headline: str,
    top_bar_color: str,
    first_name: str,
    intro: str,
    project_name: str,
    badge_style: str,
    status_label: str,
    time_label: str,
    timestamp: str,
    extra_block: str,
    cta_block: str,
    footnote: str,
) -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
        '<body style="margin:0;padding:0;background:#f4f4f7;'
        'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;">'
        '<table width="100%" cellpadding="0" cellspacing="0" '
        'style="background:#f4f4f7;padding:32px 16px;">'
        '<tr><td align="center">'
        '<table width="620" cellpadding="0" cellspacing="0" '
        'style="background:#ffffff;border:0.5px solid #e2e2e6;border-radius:12px;'
        'overflow:hidden;max-width:620px;width:100%;">'

        f'<tr><td style="height:5px;background:{top_bar_color};font-size:0;line-height:0;">&nbsp;</td></tr>'

        '<tr><td style="padding:24px 32px 20px;border-bottom:0.5px solid #e2e2e6;">'
        '<p style="margin:0 0 6px;font-size:11px;letter-spacing:0.12em;font-weight:500;color:#5a4fcf;">REJUVE BIOTECH</p>'
        f'<p style="margin:0;font-size:22px;font-weight:500;color:#111111;letter-spacing:-0.01em;">{headline}</p>'
        '</td></tr>'

        '<tr><td style="padding:24px 32px;">'
        f'<p style="margin:0 0 10px;font-size:14px;color:#111111;line-height:1.65;">Hi {first_name},</p>'
        f'<p style="margin:0 0 20px;font-size:14px;color:#555555;line-height:1.65;">{intro}</p>'

        '<table width="100%" cellpadding="0" cellspacing="0" '
        'style="background:#f8f8fa;border:0.5px solid #e2e2e6;border-radius:8px;margin-bottom:20px;">'

        '<tr><td style="padding:10px 16px;border-bottom:0.5px solid #e2e2e6;">'
        '<table width="100%"><tr>'
        '<td style="font-size:12px;color:#777777;">Project</td>'
        f'<td align="right" style="font-size:12px;color:#111111;font-weight:500;">{project_name}</td>'
        '</tr></table></td></tr>'

        '<tr><td style="padding:10px 16px;border-bottom:0.5px solid #e2e2e6;">'
        '<table width="100%"><tr>'
        '<td style="font-size:12px;color:#777777;">Status</td>'
        f'<td align="right"><span style="font-size:11px;font-weight:500;padding:3px 10px;border-radius:6px;{badge_style}">{status_label}</span></td>'
        '</tr></table></td></tr>'

        '<tr><td style="padding:10px 16px;">'
        '<table width="100%"><tr>'
        f'<td style="font-size:12px;color:#777777;">{time_label}</td>'
        f'<td align="right" style="font-size:12px;color:#111111;">{timestamp}</td>'
        '</tr></table></td></tr>'

        '</table>'

        f'{extra_block}'
        f'{cta_block}'

        f'<p style="margin:0;font-size:12px;color:#999999;line-height:1.6;">{footnote}</p>'
        '</td></tr>'

        '<tr><td style="border-top:0.5px solid #e2e2e6;padding:12px 32px;background:#f8f8fa;">'
        '<table width="100%"><tr>'
        '<td style="font-size:11px;color:#999999;">Rejuve Biotech · Hypothesis Generation</td>'
        '<td align="right" style="font-size:11px;color:#999999;">© 2026</td>'
        '</tr></table></td></tr>'

        '</table></td></tr></table></body></html>'
    )


def _cta(url: str, label: str) -> str:
    return (
        f'<a href="{url}" style="display:inline-block;background:#5a4fcf;color:#ffffff;'
        'text-decoration:none;padding:10px 22px;border-radius:8px;'
        f'font-size:13px;font-weight:500;margin-bottom:20px;">{label}</a>'
    )


def _message_block(message: str) -> str:
    return (
        '<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px;">'
        '<tr><td style="border-left:3px solid #e24b4a;padding:14px 18px;background:#fff8f8;border-radius:0 6px 6px 0;">'
        '<p style="margin:0 0 4px;font-size:11px;font-weight:600;color:#a32d2d;'
        'text-transform:uppercase;letter-spacing:0.08em;">What went wrong</p>'
        f'<p style="margin:0;font-size:13px;color:#333333;line-height:1.65;">{message}</p>'
        '</td></tr></table>'
    )


def _get_user_friendly_error(raw: str | None) -> str:
    if not raw:
        return "An unexpected error occurred during analysis. Please contact your administrator if the issue persists."

    msg = raw.lower()

    if "harmoniz" in msg or "nextflow" in msg:
        return (
            "The pipeline failed during GWAS harmonization. "
            "Please check that your input file is correctly formatted and try again."
        )
    if "filter" in msg or "significant" in msg:
        return (
            "No genome-wide significant variants were found after filtering. "
            "Your GWAS file may not contain variants that pass the significance threshold."
        )
    if "cojo" in msg or "independent signal" in msg or "out-of-bounds" in msg or "indexer" in msg:
        return (
            "No independent signals were found after variant filtering. "
            "Try adjusting your MAF threshold or significance cutoff."
        )
    if "fine" in msg or "susie" in msg or "finemap" in msg or "fine-mapping" in msg:
        return (
            "The fine-mapping step did not complete. "
            "This may be due to insufficient variants in one or more genomic regions."
        )
    if "ldsc" in msg or "heritability" in msg or "tissue" in msg:
        return (
            "The tissue enrichment analysis (LDSC) did not complete successfully. "
            "This may be a temporary issue with the reference data."
        )
    if "memory" in msg or "oom" in msg or "killed" in msg or "timeout" in msg:
        return "The pipeline timed out. This may be a temporary issue — please try again."
    if "storage" in msg or "minio" in msg or "download" in msg or "upload" in msg:
        return (
            "There was a problem accessing or downloading your GWAS file. "
            "Please verify the file is available and try again."
        )

    return "An unexpected error occurred during analysis. Please contact your administrator if the issue persists."


def build_complete_email(
    to: str,
    project_name: str,
    project_id: str,
    app_base_url: str,
    first_name: str = "there",
) -> EmailPayload:
    result_url = f"{app_base_url.rstrip('/')}/hypothesis/{project_id}"
    ts = _now_utc()

    html = _build_html(
        headline="Analysis complete",
        top_bar_color="#5a4fcf",
        first_name=first_name,
        intro=(
            f"Your analysis for project <strong style='font-weight:500;color:#111111;'>"
            f"{project_name}</strong> has finished running successfully. "
            "Your results are ready to review."
        ),
        project_name=project_name,
        badge_style="background:#eaf3de;color:#3b6d11;",
        status_label="Completed",
        time_label="Finished at",
        timestamp=ts,
        extra_block="",
        cta_block=_cta(result_url, "View results →"),
        footnote=(
            "You're receiving this because you triggered an analysis run on the "
            "Hypothesis Generation platform. If you didn't expect this, you can ignore it."
        ),
    )

    plain = (
        f"REJUVE BIOTECH — Analysis complete\n"
        f"{'─' * 40}\n\n"
        f"Hi {first_name},\n\n"
        f"Your analysis for '{project_name}' completed successfully.\n\n"
        f"Project : {project_name}\n"
        f"Status  : Completed\n"
        f"Time    : {ts}\n\n"
        f"View results: {result_url}\n\n"
        "Rejuve Biotech · Hypothesis Generation Platform"
    )

    return EmailPayload(
        to=to,
        subject=f"Rejuve Biotech | Analysis complete — {project_name}",
        body_html=html,
        body_text=plain,
    )


def build_failed_email(
    to: str,
    project_name: str,
    error: Optional[str] = None,
    first_name: str = "there",
) -> EmailPayload:
    ts = _now_utc()
    friendly = _get_user_friendly_error(error)

    html = _build_html(
        headline="Analysis failed",
        top_bar_color="#e24b4a",
        first_name=first_name,
        intro=(
            f"Unfortunately the analysis for project <strong style='font-weight:500;color:#111111;'>"
            f"{project_name}</strong> did not complete successfully."
        ),
        project_name=project_name,
        badge_style="background:#fcebeb;color:#a32d2d;",
        status_label="Failed",
        time_label="Failed at",
        timestamp=ts,
        extra_block=_message_block(friendly),
        cta_block="",
        footnote="Please retry the analysis or contact your administrator if the issue persists.",
    )

    plain = (
        f"REJUVE BIOTECH — Analysis failed\n"
        f"{'─' * 40}\n\n"
        f"Hi {first_name},\n\n"
        f"The analysis for project '{project_name}' did not complete.\n\n"
        f"Project : {project_name}\n"
        f"Status  : Failed\n"
        f"Time    : {ts}\n"
        f"Details : {friendly}\n\n"
        "Please retry or contact your administrator.\n\n"
        "Rejuve Biotech · Hypothesis Generation Platform"
    )

    return EmailPayload(
        to=to,
        subject=f"Rejuve Biotech | Analysis failed — {project_name}",
        body_html=html,
        body_text=plain,
    )
