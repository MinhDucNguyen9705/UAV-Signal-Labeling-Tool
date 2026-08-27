import io
import json
import zipfile
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from backend.rf_processor import RFProcessor

class COCOBDWExporter:
    """
    Exports RF Spectrogram annotations into standard COCO JSON format
    enriched with BDW (Burst / Waveform Description) signal parameters,
    and generates full ZIP download bundles with spectrograms, annotations, and optional .iq chunks.
    """

    @staticmethod
    def build_coco_json(rf_processor: RFProcessor,
                        classes: List[Dict[str, Any]],
                        annotations_by_chunk: Dict[int, List[Dict[str, Any]]],
                        img_width: int = 1024,
                        img_height: int = 512) -> Dict[str, Any]:
        """
        Builds COCO format dictionary containing RF session metadata and BDW parameters.
        """
        summary = rf_processor.get_summary()
        now_str = datetime.utcnow().isoformat() + "Z"

        # 1. Info & Session Parameters
        info = {
            "description": "RF Spectrogram Dataset with BDW Signal Parameters",
            "url": "https://github.com/google/antigravity",
            "version": "1.0.0",
            "year": datetime.utcnow().year,
            "contributor": "CVAT RF Spectrogram Labelling Tool",
            "date_created": now_str,
            "session_parameters": {
                "data_source": summary["source_filename"],
                "source_format": summary["source_format"],
                "fs_hz": summary["fs_hz"],
                "fs_mhz": summary["fs_mhz"],
                "center_freq_hz": summary["center_freq_hz"],
                "center_freq_mhz": summary["center_freq_mhz"],
                "total_samples": summary["total_samples"],
                "total_duration_ms": summary["total_duration_ms"],
                "total_duration_us": summary["total_duration_us"],
                "chunk_duration_ms": summary["chunk_duration_ms"],
                "num_chunks": summary["num_chunks"],
                "stft_config": summary["stft_config"]
            }
        }

        # 2. Categories
        categories = []
        for cat in classes:
            categories.append({
                "id": int(cat.get("id", 1)),
                "name": str(cat.get("name", "Signal")),
                "supercategory": "RF_Signal",
                "color": str(cat.get("color", "#00e5ff")),
                "type_of_signal": str(cat.get("type_of_signal", cat.get("name", "Unknown"))),
                "protocol": str(cat.get("protocol", "Generic"))
            })

        # Category lookup
        cat_map = {c["id"]: c for c in categories}

        # 3. Images (Chunks)
        images = []
        for chunk in rf_processor.chunks_meta:
            iq_fname = f"{chunk['file_name'].rsplit('.', 1)[0]}.iq"
            images.append({
                "id": chunk["id"] + 1,  # 1-indexed for standard COCO
                "chunk_index": chunk["id"],
                "file_name": chunk["file_name"],
                "iq_file_name": iq_fname,
                "width": img_width,
                "height": img_height,
                "time_start_us": chunk["start_time_us"],
                "time_end_us": chunk["end_time_us"],
                "duration_ms": chunk["duration_ms"],
                "freq_min_mhz": chunk["freq_min_mhz"],
                "freq_max_mhz": chunk["freq_max_mhz"],
                "fs_mhz": chunk["fs_mhz"],
                "center_freq_mhz": chunk["center_freq_mhz"],
                "start_idx": chunk.get("start_idx", 0),
                "end_idx": chunk.get("end_idx", 0)
            })

        # 4. Annotations with BDW structure
        coco_annotations = []
        annotation_counter = 1

        for chunk_id_str, box_list in annotations_by_chunk.items():
            chunk_id = int(chunk_id_str)
            if chunk_id < 0 or chunk_id >= len(rf_processor.chunks_meta):
                continue

            image_id = chunk_id + 1

            for box in box_list:
                cat_id = int(box.get("category_id", 1))
                cat_info = cat_map.get(cat_id, {
                    "name": "Unknown",
                    "type_of_signal": "Unknown",
                    "protocol": "Generic"
                })

                x = float(box["x"])
                y = float(box["y"])
                w = float(box["width"])
                h = float(box["height"])

                # COCO polygon segmentation
                segmentation = [[
                    x, y,
                    x + w, y,
                    x + w, y + h,
                    x, y + h
                ]]

                # Calculate or use existing BDW parameters
                if "bdw" in box and box["bdw"]:
                    bdw_params = box["bdw"]
                else:
                    bdw_params = rf_processor.calculate_bdw_parameters(
                        chunk_id=chunk_id,
                        bbox=[x, y, w, h],
                        img_width=img_width,
                        img_height=img_height,
                        signal_type=cat_info["type_of_signal"],
                        protocol=cat_info["protocol"]
                    )

                coco_annotations.append({
                    "id": annotation_counter,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "segmentation": segmentation,
                    "iscrowd": 0,
                    "bdw": {
                        "toa_us": bdw_params.get("toa_us", 0.0),
                        "tod_us": bdw_params.get("tod_us", 0.0),
                        "pw_us": bdw_params.get("pw_us", 0.0),
                        "fc_mhz": bdw_params.get("fc_mhz", 0.0),
                        "bw_mhz": bdw_params.get("bw_mhz", 0.0),
                        "freq_low_mhz": bdw_params.get("freq_low_mhz", 0.0),
                        "freq_high_mhz": bdw_params.get("freq_high_mhz", 0.0),
                        "snr_db": bdw_params.get("snr_db", 0.0),
                        "type_of_signal": bdw_params.get("type_of_signal", cat_info["type_of_signal"]),
                        "protocol": bdw_params.get("protocol", cat_info["protocol"]),
                        "data_source": bdw_params.get("data_source", summary["source_filename"])
                    }
                })
                annotation_counter += 1

        return {
            "info": info,
            "licenses": [{"id": 1, "name": "Antigravity RF Dataset License", "url": ""}],
            "categories": categories,
            "images": images,
            "annotations": coco_annotations
        }

    @staticmethod
    def generate_zip_bundle(rf_processor: RFProcessor,
                            classes: List[Dict[str, Any]],
                            annotations_by_chunk: Dict[int, List[Dict[str, Any]]],
                            img_width: int = 1024,
                            img_height: int = 512,
                            include_iq: bool = False) -> bytes:
        """
        Creates a full ZIP download containing spectrogram PNG images,
        COCO JSON with BDW parameters, CSV summary report, and optionally
        chunked raw .iq binary files.
        """
        coco_data = COCOBDWExporter.build_coco_json(
            rf_processor=rf_processor,
            classes=classes,
            annotations_by_chunk=annotations_by_chunk,
            img_width=img_width,
            img_height=img_height
        )

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Write COCO JSON
            coco_json_str = json.dumps(coco_data, indent=2)
            zip_file.writestr("annotations/annotations_coco.json", coco_json_str)

            # 2. Write Metadata JSON
            session_params = dict(coco_data["info"]["session_parameters"])
            session_params["include_iq_chunks"] = include_iq
            metadata_json_str = json.dumps(session_params, indent=2)
            zip_file.writestr("metadata.json", metadata_json_str)

            # 3. Write CSV BDW summary
            csv_buf = io.StringIO()
            csv_writer = csv.writer(csv_buf)
            csv_writer.writerow([
                "Annotation_ID", "Image_File", "IQ_File", "Chunk_ID", "Category", "Type_of_Signal",
                "Protocol", "TOA_us", "TOD_us", "PW_us", "FC_MHz", "BW_MHz",
                "Freq_Low_MHz", "Freq_High_MHz", "SNR_dB", "BBox_X", "BBox_Y", "BBox_W", "BBox_H"
            ])

            cat_map = {c["id"]: c["name"] for c in coco_data["categories"]}
            img_map = {img["id"]: img["file_name"] for img in coco_data["images"]}
            iq_map = {img["id"]: img.get("iq_file_name", "") for img in coco_data["images"]}

            for ann in coco_data["annotations"]:
                bdw = ann["bdw"]
                bbox = ann["bbox"]
                csv_writer.writerow([
                    ann["id"],
                    img_map.get(ann["image_id"], ""),
                    iq_map.get(ann["image_id"], ""),
                    ann["image_id"] - 1,
                    cat_map.get(ann["category_id"], "Unknown"),
                    bdw.get("type_of_signal", ""),
                    bdw.get("protocol", ""),
                    bdw.get("toa_us", ""),
                    bdw.get("tod_us", ""),
                    bdw.get("pw_us", ""),
                    bdw.get("fc_mhz", ""),
                    bdw.get("bw_mhz", ""),
                    bdw.get("freq_low_mhz", ""),
                    bdw.get("freq_high_mhz", ""),
                    bdw.get("snr_db", ""),
                    bbox[0], bbox[1], bbox[2], bbox[3]
                ])

            zip_file.writestr("signal_parameters_bdw.csv", csv_buf.getvalue())

            # 4. Render and write all Spectrogram Images
            for chunk in rf_processor.chunks_meta:
                c_id = chunk["id"]
                img_bytes, _ = rf_processor.render_spectrogram_image(
                    chunk_id=c_id,
                    width=img_width,
                    height=img_height
                )
                zip_file.writestr(f"images/{chunk['file_name']}", img_bytes)

            # 5. Optionally write raw .iq chunks
            if include_iq and rf_processor.iq_data is not None:
                for chunk in rf_processor.chunks_meta:
                    start_idx = chunk["start_idx"]
                    end_idx = chunk["end_idx"]
                    chunk_iq = rf_processor.iq_data[start_idx:end_idx]
                    # Convert to complex64 bytes (float32 real + float32 imag interleaved)
                    iq_bytes = chunk_iq.astype(np.complex64).tobytes()
                    iq_filename = f"{chunk['file_name'].rsplit('.', 1)[0]}.iq"
                    zip_file.writestr(f"iq/{iq_filename}", iq_bytes)

        return zip_buffer.getvalue()

class COCOBDWImporter:
    """
    Parses and imports standard COCO format JSON files containing annotations,
    categories, and optional BDW parameters into the application session.
    """
    DEFAULT_PALETTE = ['#00e5ff', '#ff007f', '#00ff66', '#ffaa00', '#bd00ff', '#ff3333', '#ffff00', '#00e1d9']

    @staticmethod
    def parse_coco_json(coco_data: Dict[str, Any],
                        rf_processor: RFProcessor,
                        existing_classes: List[Dict[str, Any]],
                        img_width: int = 1024,
                        img_height: int = 512) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parses COCO JSON and returns (annotations_by_chunk, updated_classes, stats).
        """
        # 1. Parse & Merge Categories
        classes = [dict(c) for c in existing_classes]
        existing_names = {c["name"].lower(): c for c in classes}
        existing_ids = {int(c["id"]): c for c in classes}
        cat_id_mapping = {}  # incoming_cat_id -> session_cat_id

        incoming_cats = coco_data.get("categories", [])
        for inc_cat in incoming_cats:
            inc_id = int(inc_cat.get("id", len(cat_id_mapping) + 1))
            inc_name = str(inc_cat.get("name", f"Class_{inc_id}"))

            if inc_name.lower() in existing_names:
                match = existing_names[inc_name.lower()]
                cat_id_mapping[inc_id] = match["id"]
            elif inc_id in existing_ids:
                cat_id_mapping[inc_id] = inc_id
            else:
                next_id = max([c["id"] for c in classes], default=0) + 1
                color = inc_cat.get("color") or COCOBDWImporter.DEFAULT_PALETTE[(next_id - 1) % len(COCOBDWImporter.DEFAULT_PALETTE)]
                new_cls = {
                    "id": next_id,
                    "name": inc_name,
                    "color": color,
                    "type_of_signal": str(inc_cat.get("type_of_signal", inc_name)),
                    "protocol": str(inc_cat.get("protocol", "Generic"))
                }
                classes.append(new_cls)
                existing_names[inc_name.lower()] = new_cls
                existing_ids[next_id] = new_cls
                cat_id_mapping[inc_id] = next_id

        # 2. Build Image ID to Chunk Index map
        img_id_to_chunk = {}
        total_chunks = len(rf_processor.chunks_meta)

        for img in coco_data.get("images", []):
            img_id = img.get("id")
            if "chunk_index" in img:
                chunk_id = int(img["chunk_index"])
            elif "file_name" in img:
                fname = img["file_name"]
                matched_chunk = next((c for c in rf_processor.chunks_meta if c["file_name"] == fname), None)
                if matched_chunk:
                    chunk_id = matched_chunk["id"]
                else:
                    import re
                    m = re.search(r'chunk_?(\d+)', fname, re.IGNORECASE)
                    if m:
                        chunk_id = int(m.group(1))
                    elif isinstance(img_id, int):
                        chunk_id = img_id - 1
                    else:
                        chunk_id = 0
            elif isinstance(img_id, int):
                chunk_id = img_id - 1
            else:
                chunk_id = 0

            img_id_to_chunk[img_id] = chunk_id

        # 3. Parse Annotations
        annotations_by_chunk: Dict[str, List[Dict[str, Any]]] = {}
        total_imported = 0

        for ann in coco_data.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id in img_id_to_chunk:
                chunk_id = img_id_to_chunk[img_id]
            elif isinstance(img_id, int) and (img_id - 1) < total_chunks:
                chunk_id = img_id - 1
            else:
                chunk_id = 0

            if chunk_id < 0 or (total_chunks > 0 and chunk_id >= total_chunks):
                continue

            chunk_id_str = str(chunk_id)
            if chunk_id_str not in annotations_by_chunk:
                annotations_by_chunk[chunk_id_str] = []

            bbox = ann.get("bbox", [0, 0, 50, 50])
            if len(bbox) >= 4:
                x = float(bbox[0])
                y = float(bbox[1])
                w = float(bbox[2])
                h = float(bbox[3])
            else:
                continue

            inc_cat_id = ann.get("category_id", 1)
            target_cat_id = cat_id_mapping.get(inc_cat_id, 1)
            target_cat = next((c for c in classes if c["id"] == target_cat_id), None)

            # BDW parameters
            bdw = ann.get("bdw")
            if not bdw or not isinstance(bdw, dict):
                sig_type = target_cat["type_of_signal"] if target_cat else "Unknown"
                proto = target_cat["protocol"] if target_cat else "Generic"
                bdw = rf_processor.calculate_bdw_parameters(
                    chunk_id=chunk_id,
                    bbox=[x, y, w, h],
                    img_width=img_width,
                    img_height=img_height,
                    signal_type=sig_type,
                    protocol=proto
                )

            box_obj = {
                "id": f"box_{ann.get('id', total_imported + 1)}",
                "category_id": target_cat_id,
                "x": round(x, 2),
                "y": round(y, 2),
                "width": round(w, 2),
                "height": round(h, 2),
                "isLocked": False,
                "isHidden": False,
                "bdw": bdw
            }
            annotations_by_chunk[chunk_id_str].append(box_obj)
            total_imported += 1

        stats = {
            "total_imported": total_imported,
            "chunks_updated": len(annotations_by_chunk),
            "classes_count": len(classes)
        }

        return annotations_by_chunk, classes, stats
