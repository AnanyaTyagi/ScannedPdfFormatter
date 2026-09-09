#!/usr/bin/env python3
"""
run_full_evaluation.py
ONE-COMMAND automated evaluation of entire PDF tagging pipeline

Usage:
    python run_full_evaluation.py --test-pdfs ./test_pdfs --ground-truth ./ground_truth

This will:
1. Process all test PDFs through the pipeline
2. Calculate OCR accuracy automatically
3. Analyze ML vs heuristic statistics
4. Measure processing speed
5. Generate comprehensive report
"""

import os
import sys
import json
import time
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import glob

try:
    import Levenshtein
except ImportError:
    print("Installing python-Levenshtein...")
    subprocess.run([sys.executable, "-m", "pip", "install", "python-Levenshtein", "-q"])
    import Levenshtein


class AutomatedEvaluator:
    def __init__(self, test_pdfs_dir, ground_truth_dir=None, output_dir="evaluation_results"):
        self.test_pdfs_dir = Path(test_pdfs_dir)
        self.ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Find scripts automatically
        self.script_dir = Path(__file__).parent
        self.tagmypdf_script = self._find_script("TagMyPDF2.py")

        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_pdfs': [],
            'ocr_accuracy': None,
            'tagging_stats': None,
            'processing_times': [],
            'overall_summary': {}
        }

        print(f"📁 Test PDFs: {self.test_pdfs_dir}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🔧 Script: {self.tagmypdf_script}")

    def _find_script(self, script_name):
        """Find script in current directory"""
        script_path = self.script_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Cannot find {script_name}")
        return script_path

    def run_complete_evaluation(self):
        """
        Main evaluation workflow - processes all PDFs and generates report
        """
        print("\n" + "=" * 70)
        print("🚀 STARTING AUTOMATED PIPELINE EVALUATION")
        print("=" * 70)

        # Step 1: Get list of test PDFs
        pdf_files = list(self.test_pdfs_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {self.test_pdfs_dir}")
            sys.exit(1)

        print(f"\n📄 Found {len(pdf_files)} PDF(s) to process\n")

        # Step 2: Process each PDF through pipeline
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n{'=' * 70}")
            print(f"Processing {idx}/{len(pdf_files)}: {pdf_path.name}")
            print(f"{'=' * 70}")

            result = self._process_single_pdf(pdf_path)
            self.results['test_pdfs'].append(result)

        # Step 3: Aggregate statistics
        self._aggregate_statistics()

        # Step 4: Calculate OCR accuracy (if ground truth provided)
        if self.ground_truth_dir and self.ground_truth_dir.exists():
            self._calculate_ocr_accuracy()

        # Step 5: Generate reports
        self._generate_reports()

        print(f"\n{'=' * 70}")
        print("✅ EVALUATION COMPLETE!")
        print(f"{'=' * 70}")
        print(f"📊 Results: {self.output_dir / 'EVALUATION_REPORT.md'}")

    def _process_single_pdf(self, pdf_path):
        """
        Process one PDF through the entire pipeline and collect metrics
        """
        pdf_name = pdf_path.name

        # Create working directory for this PDF
        work_dir = self.output_dir / f"work_{pdf_path.stem}"
        work_dir.mkdir(exist_ok=True)

        # Copy PDF to work directory
        work_pdf = work_dir / "input.pdf"
        shutil.copy2(pdf_path, work_pdf)

        result = {
            'pdf_name': pdf_name,
            'pdf_path': str(pdf_path),
            'work_dir': str(work_dir),
            'success': False,
            'error': None,
            'timing': {},
            'stats': {}
        }

        try:
            # Get page count
            import fitz
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            result['page_count'] = page_count

            print(f"📄 Pages: {page_count}")

            # Run TagMyPDF2.py with timing
            start_time = time.time()

            print(f"⏳ Running pipeline...")
            proc_result = subprocess.run(
                [sys.executable, str(self.tagmypdf_script), "input.pdf"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout per PDF
            )

            end_time = time.time()
            elapsed = end_time - start_time

            result['timing'] = {
                'total_seconds': round(elapsed, 2),
                'seconds_per_page': round(elapsed / page_count, 2),
                'minutes': round(elapsed / 60, 2)
            }

            if proc_result.returncode == 0:
                result['success'] = True
                print(f"✅ Success in {elapsed:.1f}s ({elapsed / page_count:.1f}s/page)")

                # Collect statistics from output files
                result['stats'] = self._collect_pdf_statistics(work_dir)

            else:
                result['error'] = f"Exit code {proc_result.returncode}"
                print(f"❌ Failed: {result['error']}")
                if proc_result.stderr:
                    print(f"   Error: {proc_result.stderr[-200:]}")

        except subprocess.TimeoutExpired:
            result['error'] = "Timeout (>10 minutes)"
            print(f"❌ Timeout")

        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Error: {e}")

        return result

    def _collect_pdf_statistics(self, work_dir):
        """
        Collect statistics from pipeline output files
        """
        stats = {
            'total_tags': 0,
            'ml_tags': 0,
            'heuristic_tags': 0,
            'artifact_tags': 0,
            'tag_sources': {},
            'tag_types': {},
            'heuristic_breakdown': {}
        }

        # Read fused_tags/*.tags.json files
        fused_dir = work_dir / "fused_tags"
        if not fused_dir.exists():
            return stats

        for tag_file in fused_dir.glob("page_*.tags.json"):
            with open(tag_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tags = data.get('tags', [])

            for tag_obj in tags:
                tag_type = tag_obj.get('tag', 'unknown')
                source = tag_obj.get('source', 'unknown')

                stats['total_tags'] += 1

                # Count tag types
                stats['tag_types'][tag_type] = stats['tag_types'].get(tag_type, 0) + 1

                # Count sources
                stats['tag_sources'][source] = stats['tag_sources'].get(source, 0) + 1

                # Categorize
                if source == "artifact_detection":
                    stats['artifact_tags'] += 1
                elif source.startswith("ML:"):
                    stats['ml_tags'] += 1
                elif source.startswith("heuristic:"):
                    stats['heuristic_tags'] += 1
                    # Extract heuristic type
                    h_type = source.replace("heuristic:", "")
                    stats['heuristic_breakdown'][h_type] = \
                        stats['heuristic_breakdown'].get(h_type, 0) + 1
                elif source == "default":
                    stats['heuristic_tags'] += 1

        return stats

    def _aggregate_statistics(self):
        """
        Aggregate statistics across all PDFs
        """
        print(f"\n{'=' * 70}")
        print("📊 AGGREGATING STATISTICS")
        print(f"{'=' * 70}")

        successful = [r for r in self.results['test_pdfs'] if r['success']]
        failed = [r for r in self.results['test_pdfs'] if not r['success']]

        if not successful:
            print("❌ No PDFs processed successfully")
            return

        # Aggregate timing
        total_pages = sum(r['page_count'] for r in successful)
        total_time = sum(r['timing']['total_seconds'] for r in successful)
        avg_time_per_page = total_time / total_pages if total_pages > 0 else 0

        # Aggregate tagging stats
        total_tags = sum(r['stats']['total_tags'] for r in successful)
        total_ml = sum(r['stats']['ml_tags'] for r in successful)
        total_heuristic = sum(r['stats']['heuristic_tags'] for r in successful)
        total_artifact = sum(r['stats']['artifact_tags'] for r in successful)

        # Aggregate heuristic breakdown
        heuristic_breakdown = {}
        for r in successful:
            for h_type, count in r['stats']['heuristic_breakdown'].items():
                heuristic_breakdown[h_type] = heuristic_breakdown.get(h_type, 0) + count

        # Aggregate tag types
        tag_types = {}
        for r in successful:
            for tag_type, count in r['stats']['tag_types'].items():
                tag_types[tag_type] = tag_types.get(tag_type, 0) + count

        self.results['overall_summary'] = {
            'total_pdfs': len(self.results['test_pdfs']),
            'successful_pdfs': len(successful),
            'failed_pdfs': len(failed),
            'total_pages': total_pages,
            'total_processing_time': round(total_time, 2),
            'avg_seconds_per_page': round(avg_time_per_page, 2),
            'avg_seconds_per_pdf': round(total_time / len(successful), 2),
            'tagging': {
                'total_tags': total_tags,
                'ml_tags': total_ml,
                'heuristic_tags': total_heuristic,
                'artifact_tags': total_artifact,
                'ml_percent': round(total_ml / total_tags * 100, 2) if total_tags > 0 else 0,
                'heuristic_percent': round(total_heuristic / total_tags * 100, 2) if total_tags > 0 else 0,
                'artifact_percent': round(total_artifact / total_tags * 100, 2) if total_tags > 0 else 0,
                'heuristic_breakdown': heuristic_breakdown,
                'tag_types': tag_types
            }
        }

        # Print summary
        summary = self.results['overall_summary']
        print(f"\n✅ Processed: {summary['successful_pdfs']}/{summary['total_pdfs']} PDFs")
        print(f"📄 Total Pages: {summary['total_pages']}")
        print(f"⏱️  Average Speed: {summary['avg_seconds_per_page']:.2f} seconds/page")
        print(f"\n🏷️  Tagging Statistics:")
        print(f"   Total Tags: {summary['tagging']['total_tags']}")
        print(f"   ML-based: {summary['tagging']['ml_tags']} ({summary['tagging']['ml_percent']:.1f}%)")
        print(f"   Heuristic: {summary['tagging']['heuristic_tags']} ({summary['tagging']['heuristic_percent']:.1f}%)")
        print(f"   Artifacts: {summary['tagging']['artifact_tags']} ({summary['tagging']['artifact_percent']:.1f}%)")

    def _calculate_ocr_accuracy(self):
        """
        Calculate OCR accuracy using ground truth files
        """
        print(f"\n{'=' * 70}")
        print("📝 CALCULATING OCR ACCURACY")
        print(f"{'=' * 70}")

        if not self.ground_truth_dir:
            print("⚠️  No ground truth directory provided, skipping OCR accuracy")
            return

        gt_files = list(self.ground_truth_dir.glob("*.txt"))
        if not gt_files:
            print(f"⚠️  No ground truth .txt files found in {self.ground_truth_dir}")
            return

        print(f"Found {len(gt_files)} ground truth files")

        page_results = []

        for gt_file in gt_files:
            # Load ground truth
            with open(gt_file, 'r', encoding='utf-8') as f:
                gt_text = f.read()

            # Find corresponding OCR output
            # Match page_001.txt -> work_*/lines_out/page_001.lines.json
            page_name = gt_file.stem

            ocr_text = None
            for pdf_result in self.results['test_pdfs']:
                if not pdf_result['success']:
                    continue

                work_dir = Path(pdf_result['work_dir'])
                lines_file = work_dir / "lines_out" / f"{page_name}.lines.json"

                if lines_file.exists():
                    with open(lines_file, 'r', encoding='utf-8') as f:
                        lines_data = json.load(f)
                    ocr_text = '\n'.join([line['text'] for line in lines_data.get('lines', [])])
                    break

            if ocr_text:
                # Calculate metrics
                metrics = self._calculate_ocr_metrics(gt_text, ocr_text)
                metrics['page'] = page_name
                page_results.append(metrics)

                print(
                    f"  {page_name}: CER={metrics['cer']:.2f}%, WER={metrics['wer']:.2f}%, Acc={metrics['accuracy']:.2f}%")

        if page_results:
            avg_cer = sum(r['cer'] for r in page_results) / len(page_results)
            avg_wer = sum(r['wer'] for r in page_results) / len(page_results)
            avg_acc = sum(r['accuracy'] for r in page_results) / len(page_results)

            self.results['ocr_accuracy'] = {
                'pages': page_results,
                'summary': {
                    'avg_cer': round(avg_cer, 2),
                    'avg_wer': round(avg_wer, 2),
                    'avg_accuracy': round(avg_acc, 2),
                    'pages_evaluated': len(page_results)
                }
            }

            print(f"\n📊 OCR Accuracy Summary:")
            print(f"   Average CER: {avg_cer:.2f}%")
            print(f"   Average WER: {avg_wer:.2f}%")
            print(f"   Average Accuracy: {avg_acc:.2f}%")

    def _calculate_ocr_metrics(self, gt_text, ocr_text):
        """Calculate CER, WER, Accuracy"""

        def normalize(text):
            import re
            text = re.sub(r'\s+', ' ', text)
            return text.strip().lower()

        gt_norm = normalize(gt_text)
        ocr_norm = normalize(ocr_text)

        # CER
        cer_dist = Levenshtein.distance(gt_norm, ocr_norm)
        cer = (cer_dist / len(gt_norm) * 100) if len(gt_norm) > 0 else 0

        # WER
        gt_words = gt_norm.split()
        ocr_words = ocr_norm.split()
        wer_dist = Levenshtein.distance(' '.join(gt_words), ' '.join(ocr_words))
        wer = (wer_dist / len(gt_words) * 100) if len(gt_words) > 0 else 0

        # Accuracy
        accuracy = 100 - cer

        return {'cer': cer, 'wer': wer, 'accuracy': accuracy}

    def _generate_reports(self):
        """
        Generate JSON and Markdown reports
        """
        print(f"\n{'=' * 70}")
        print("📄 GENERATING REPORTS")
        print(f"{'=' * 70}")

        # Save JSON
        json_file = self.output_dir / "evaluation_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ JSON report: {json_file}")

        # Generate Markdown
        md_file = self.output_dir / "EVALUATION_REPORT.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            self._write_markdown_report(f)
        print(f"✓ Markdown report: {md_file}")

        # Generate Code4Lib-ready summary
        code4lib_file = self.output_dir / "CODE4LIB_METRICS.md"
        with open(code4lib_file, 'w', encoding='utf-8') as f:
            self._write_code4lib_summary(f)
        print(f"✓ Code4Lib metrics: {code4lib_file}")

    def _write_markdown_report(self, f):
        """Write comprehensive markdown report"""
        f.write("# PDF Tagging Pipeline Evaluation Report\n\n")
        f.write(f"**Generated:** {self.results['timestamp']}\n\n")
        f.write("---\n\n")

        summary = self.results['overall_summary']

        # Overview
        f.write("## 📊 Overview\n\n")
        f.write(f"- **Total PDFs Tested:** {summary['total_pdfs']}\n")
        f.write(f"- **Successfully Processed:** {summary['successful_pdfs']}\n")
        f.write(f"- **Failed:** {summary['failed_pdfs']}\n")
        f.write(f"- **Total Pages:** {summary['total_pages']}\n")
        f.write(
            f"- **Total Processing Time:** {summary['total_processing_time']:.1f} seconds ({summary['total_processing_time'] / 60:.1f} minutes)\n\n")

        # Processing Speed
        f.write("## ⏱️ Processing Speed\n\n")
        f.write(f"- **Average per Page:** {summary['avg_seconds_per_page']:.2f} seconds\n")
        f.write(f"- **Average per PDF:** {summary['avg_seconds_per_pdf']:.2f} seconds\n")
        f.write(f"- **Throughput:** {60 / summary['avg_seconds_per_page']:.1f} pages/minute\n\n")

        # Individual PDF Results
        f.write("### Individual PDF Results\n\n")
        f.write("| PDF | Pages | Time (s) | s/page | Status |\n")
        f.write("|-----|-------|----------|--------|--------|\n")
        for r in self.results['test_pdfs']:
            if r['success']:
                f.write(
                    f"| {r['pdf_name']} | {r['page_count']} | {r['timing']['total_seconds']:.1f} | {r['timing']['seconds_per_page']:.2f} | ✅ |\n")
            else:
                f.write(f"| {r['pdf_name']} | - | - | - | ❌ {r['error']} |\n")
        f.write("\n")

        # Tagging Statistics
        tagging = summary['tagging']
        f.write("## 🏷️ Tagging Statistics\n\n")
        f.write(f"- **Total Tags:** {tagging['total_tags']:,}\n")
        f.write(f"- **ML-based Tags:** {tagging['ml_tags']:,} ({tagging['ml_percent']:.1f}%)\n")
        f.write(f"- **Heuristic-based Tags:** {tagging['heuristic_tags']:,} ({tagging['heuristic_percent']:.1f}%)\n")
        f.write(f"- **Artifact Tags:** {tagging['artifact_tags']:,} ({tagging['artifact_percent']:.1f}%)\n\n")

        # Heuristic Breakdown
        f.write("### Heuristic Logic Breakdown\n\n")
        total_h = tagging['heuristic_tags']
        for h_type, count in sorted(tagging['heuristic_breakdown'].items(), key=lambda x: x[1], reverse=True):
            percent = (count / total_h * 100) if total_h > 0 else 0
            f.write(f"- **{h_type}:** {count:,} ({percent:.1f}%)\n")
        f.write("\n")

        # Heuristic Logic Explanations
        f.write("### Heuristic Logic Explanations\n\n")
        heuristics_explained = {
            'font_size': 'Large font size detected (>1.4x body text) → Tagged as heading',
            'font_bold': 'Bold text with moderate size increase → Tagged as H2',
            'list_pattern': 'Detected bullet points or numbering (•, 1., a.) → Tagged as list item',
            'caption_pattern': 'Text matches "Figure X:" or "Table X:" pattern → Tagged as caption',
            'h1_pattern': 'Text matches major section names (Abstract, Introduction, etc.) → Tagged as H1',
            'h2_pattern': 'Text matches numbered section patterns (1., I., A.) → Tagged as H2',
            'all_caps': 'All-caps text with size increase → Tagged as H2',
            'default': 'No specific pattern detected → Tagged as paragraph (P)'
        }

        for h_type in sorted(tagging['heuristic_breakdown'].keys()):
            if h_type in heuristics_explained:
                f.write(f"**{h_type}:** {heuristics_explained[h_type]}\n\n")

        # Tag Type Distribution
        f.write("### Tag Type Distribution\n\n")
        f.write("| Tag Type | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        for tag_type, count in sorted(tagging['tag_types'].items(), key=lambda x: x[1], reverse=True):
            percent = (count / tagging['total_tags'] * 100) if tagging['total_tags'] > 0 else 0
            f.write(f"| {tag_type} | {count:,} | {percent:.1f}% |\n")
        f.write("\n")

        # OCR Accuracy
        if self.results['ocr_accuracy']:
            ocr = self.results['ocr_accuracy']['summary']
            f.write("## 📝 OCR Accuracy\n\n")
            f.write(f"- **Character Error Rate (CER):** {ocr['avg_cer']:.2f}%\n")
            f.write(f"- **Word Error Rate (WER):** {ocr['avg_wer']:.2f}%\n")
            f.write(f"- **Overall Accuracy:** {ocr['avg_accuracy']:.2f}%\n")
            f.write(f"- **Pages Evaluated:** {ocr['pages_evaluated']}\n\n")

            f.write("### Per-Page OCR Results\n\n")
            f.write("| Page | CER | WER | Accuracy |\n")
            f.write("|------|-----|-----|----------|\n")
            for page in self.results['ocr_accuracy']['pages']:
                f.write(f"| {page['page']} | {page['cer']:.2f}% | {page['wer']:.2f}% | {page['accuracy']:.2f}% |\n")
            f.write("\n")

    def _write_code4lib_summary(self, f):
        """Write Code4Lib-ready metrics summary"""
        f.write("# Code4Lib Submission Metrics\n\n")
        f.write("*Ready-to-use statistics for your Code4Lib paper*\n\n")
        f.write("---\n\n")

        summary = self.results['overall_summary']
        tagging = summary['tagging']

        f.write("## Performance Metrics\n\n")
        f.write("```\n")
        f.write("Processing Speed:\n")
        f.write(f"- Average: {summary['avg_seconds_per_page']:.2f} seconds/page\n")
        f.write(f"- Throughput: {60 / summary['avg_seconds_per_page']:.1f} pages/minute\n")
        f.write(f"- Hardware: CPU-only (specify your CPU model)\n")
        f.write(f"- Test corpus: {summary['total_pdfs']} PDFs, {summary['total_pages']} pages\n")
        f.write("```\n\n")

        f.write("## Tagging Methodology\n\n")
        f.write("```\n")
        f.write("Tag Assignment:\n")
        f.write(f"- ML-based (YOLO layout detection): {tagging['ml_percent']:.1f}%\n")
        f.write(f"- Heuristic-based (pattern/font analysis): {tagging['heuristic_percent']:.1f}%\n")
        f.write(f"- Artifact detection (headers/footers/page numbers): {tagging['artifact_percent']:.1f}%\n")
        f.write(f"- Total tags processed: {tagging['total_tags']:,}\n")
        f.write("```\n\n")

        f.write("## Heuristic Rules Employed\n\n")
        f.write("The system uses the following heuristic rules:\n\n")
        for h_type, count in sorted(tagging['heuristic_breakdown'].items(), key=lambda x: x[1], reverse=True):
            percent = (count / tagging['heuristic_tags'] * 100) if tagging['heuristic_tags'] > 0 else 0
            f.write(f"- **{h_type}:** {count:,} instances ({percent:.1f}% of heuristic tags)\n")
        f.write("\n")

        if self.results['ocr_accuracy']:
            ocr = self.results['ocr_accuracy']['summary']
            f.write("## OCR Accuracy\n\n")
            f.write("```\n")
            f.write("OCR Performance:\n")
            f.write(f"- Character Error Rate (CER): {ocr['avg_cer']:.2f}%\n")
            f.write(f"- Word Error Rate (WER): {ocr['avg_wer']:.2f}%\n")
            f.write(f"- Overall Accuracy: {ocr['avg_accuracy']:.2f}%\n")
            f.write(f"- Evaluation: {ocr['pages_evaluated']} pages manually validated\n")
            f.write("- OCR Engine: EasyOCR (or Google Vision - specify which you used)\n")
            f.write("```\n\n")

        f.write("## Copy-Paste for Paper\n\n")
        f.write("```markdown\n")
        f.write("The system processes documents at an average rate of\n")
        f.write(f"{summary['avg_seconds_per_page']:.2f} seconds per page on CPU-only hardware.\n")
        f.write(f"Tag assignment combines ML-based layout detection ({tagging['ml_percent']:.1f}%)\n")
        f.write(f"with heuristic pattern matching ({tagging['heuristic_percent']:.1f}%).\n")

        if self.results['ocr_accuracy']:
            f.write(f"OCR accuracy was measured at {ocr['avg_accuracy']:.1f}%\n")
            f.write(f"(WER: {ocr['avg_wer']:.2f}%) across {ocr['pages_evaluated']} test pages.\n")

        f.write("```\n")


def main():
    parser = argparse.ArgumentParser(
        description="Automated PDF Tagging Pipeline Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate without OCR accuracy (no ground truth)
  python run_full_evaluation.py --test-pdfs ./test_pdfs

  # Evaluate with OCR accuracy
  python run_full_evaluation.py --test-pdfs ./test_pdfs --ground-truth ./ground_truth

  # Custom output directory
  python run_full_evaluation.py --test-pdfs ./test_pdfs --output ./results
        """
    )

    parser.add_argument('--test-pdfs', required=True,
                        help='Directory containing PDF files to test')
    parser.add_argument('--ground-truth', default=None,
                        help='Directory with ground truth .txt files for OCR accuracy (optional)')
    parser.add_argument('--output', default='evaluation_results',
                        help='Output directory for results (default: evaluation_results)')

    args = parser.parse_args()

    # Run evaluation
    evaluator = AutomatedEvaluator(
        test_pdfs_dir=args.test_pdfs,
        ground_truth_dir=args.ground_truth,
        output_dir=args.output
    )

    evaluator.run_complete_evaluation()


if __name__ == "__main__":
    main()