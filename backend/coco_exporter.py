import io
import json
import zipfile
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from backend.rf_processor import RFProcessor, add_awgn
from backend.logger import logger
import os

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
                        drone_name: Optional[str] = None,
                        img_width: int = 1024,
                        img_height: int = 512,
                        img_format: str = "png",
                        export_snr_db: Optional[float] = None) -> Dict[str, Any]:
        """
        Builds COCO format dictionary containing RF session metadata and BDW parameters.
        """
        summary = rf_processor.get_summary()
        effective_drone = drone_name or summary.get("drone_name") or "DJI-MAVIC-PRO-3"
        clean_ext = img_format.lower().lstrip('.')
        if clean_ext not in ["png", "jpg", "jpeg"]:
            clean_ext = "png"
        now_str = datetime.utcnow().isoformat() + "Z"

        # 1. Info & Session Parameters
        session_params = {
            "dataset_format": "coco",
            "drone_name": effective_drone,
            "data_source": summary["source_filename"],
            "source_format": summary["source_format"],
            "fs_hz": summary["fs_hz"],
            "fs_mhz": summary["fs_mhz"],
            "center_freq_hz": summary["center_freq_hz"],
            "center_freq_mhz": summary["center_freq_mhz"],
            "img_width": img_width,
            "img_height": img_height,
            "img_format": clean_ext,
            "total_samples": summary["total_samples"],
            "total_duration_ms": summary["total_duration_ms"],
            "total_duration_us": summary["total_duration_us"],
            "chunk_duration_ms": summary["chunk_duration_ms"],
            "num_chunks": summary["num_chunks"],
            "stft_config": summary["stft_config"]
        }
        if export_snr_db is not None:
            session_params["export_snr_db"] = float(export_snr_db)
            session_params["awgn_applied_on_export"] = True

        info = {
            "description": "RF Spectrogram Dataset with BDW Signal Parameters",
            "url": "https://github.com/google/antigravity",
            "version": "1.0.0",
            "year": datetime.utcnow().year,
            "contributor": "CVAT RF Spectrogram Labelling Tool",
            "date_created": now_str,
            "session_parameters": session_params
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
            c_id = chunk["id"]
            img_fname = rf_processor.get_formatted_filename(chunk_id=c_id, extension=clean_ext, drone_name=effective_drone)
            iq_fname = rf_processor.get_formatted_filename(chunk_id=c_id, extension="iq", drone_name=effective_drone)
            images.append({
                "id": c_id + 1,  # 1-indexed for standard COCO
                "chunk_index": c_id,
                "file_name": img_fname,
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

                box_src_w = float(box.get("img_width") or rf_processor.render_width or 1024)
                box_src_h = float(box.get("img_height") or rf_processor.render_height or 512)

                x = float(box["x"])
                y = float(box["y"])
                w = float(box["width"])
                h = float(box["height"])

                scale_x = img_width / box_src_w if box_src_w > 0 else 1.0
                scale_y = img_height / box_src_h if box_src_h > 0 else 1.0

                exp_x = round(x * scale_x, 2)
                exp_y = round(y * scale_y, 2)
                exp_w = round(w * scale_x, 2)
                exp_h = round(h * scale_y, 2)

                # COCO polygon segmentation
                segmentation = [[
                    exp_x, exp_y,
                    exp_x + exp_w, exp_y,
                    exp_x + exp_w, exp_y + exp_h,
                    exp_x, exp_y + exp_h
                ]]

                # Calculate or use existing BDW parameters
                if "bdw" in box and box["bdw"]:
                    bdw_params = box["bdw"]
                else:
                    bdw_params = rf_processor.calculate_bdw_parameters(
                        chunk_id=chunk_id,
                        bbox=[x, y, w, h],
                        img_width=int(box_src_w),
                        img_height=int(box_src_h),
                        signal_type=cat_info["type_of_signal"],
                        protocol=cat_info["protocol"]
                    )

                snr_value = float(export_snr_db) if export_snr_db is not None else bdw_params.get("snr_db", 0.0)
                coco_annotations.append({
                    "id": annotation_counter,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [exp_x, exp_y, exp_w, exp_h],
                    "area": round(exp_w * exp_h, 2),
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
                        "snr_db": round(snr_value, 1),
                        "type_of_signal": bdw_params.get("type_of_signal", cat_info["type_of_signal"]),
                        "protocol": bdw_params.get("protocol", cat_info["protocol"]),
                        "data_source": bdw_params.get("data_source", summary["source_filename"])
                    }
                })
                annotation_counter += 1

        logger.info(f"Built COCO JSON: {len(images)} images, {len(coco_annotations)} annotations across {len(categories)} categories.")
        return {
            "info": info,
            "licenses": [{"id": 1, "name": "Antigravity RF Dataset License", "url": ""}],
            "images": images,
            "annotations": coco_annotations,
            "categories": categories
        }

    @staticmethod
    def generate_zip_bundle(
        rf_processor: RFProcessor,
        classes: List[Dict[str, Any]],
        annotations_by_chunk: Dict[int, List[Dict[str, Any]]],
        drone_name: Optional[str] = None,
        img_width: int = 1024,
        img_height: int = 512,
        img_format: str = "png",
        include_images: bool = True,
        include_coco_json: bool = True,
        include_csv: bool = True,
        include_iq: bool = False,
        include_video: bool = False,
        include_metadata: bool = True,
        export_snr_db: Optional[float] = None,
        progress_callback: Optional[Callable[[int, str, str, str], None]] = None
    ) -> bytes:
        """
        Creates a complete dataset ZIP bundle containing selectable components:
        - annotations_coco_bdw.json (COCO format)
        - spectrograms/*.png (or .jpg)
        - signal_parameters_bdw.csv
        - (optional) metadata.json
        - (optional) iq/*.iq (raw IQ chunks with optional AWGN)
        - (optional) video/*.mp4 (waterfall video)
        All named according to standard format.
        """
        import concurrent.futures
        import threading

        summary = rf_processor.get_summary()
        effective_drone = drone_name or summary.get("drone_name") or "DJI-MAVIC-PRO-3"
        clean_ext = img_format.lower().lstrip('.')
        if clean_ext not in ["jpg", "jpeg", "png"]:
            clean_ext = "png"

        if progress_callback:
            progress_callback(2, "metadata", "Initializing COCO export package...", "Preparing annotations JSON & metadata")

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Add COCO JSON if selected
            coco_json_dict = COCOBDWExporter.build_coco_json(
                rf_processor=rf_processor,
                classes=classes,
                annotations_by_chunk=annotations_by_chunk,
                export_snr_db=export_snr_db,
                drone_name=effective_drone,
                img_width=img_width,
                img_height=img_height,
                img_format=clean_ext
            )
            if include_coco_json:
                coco_json_str = json.dumps(coco_json_dict, indent=2)
                zip_file.writestr("annotations_coco_bdw.json", coco_json_str)

            # 2. Add Metadata JSON if selected
            if include_metadata:
                session_params = dict(coco_json_dict.get("info", {}).get("session_parameters", {}))
                session_params["include_images"] = include_images
                session_params["include_coco_json"] = include_coco_json
                session_params["include_csv"] = include_csv
                session_params["include_iq_chunks"] = include_iq
                session_params["include_video"] = include_video
                if export_snr_db is not None:
                    session_params["export_snr_db"] = float(export_snr_db)
                    session_params["awgn_applied_on_export"] = True
                zip_file.writestr("metadata.json", json.dumps(session_params, indent=2))

            # 3. Add CSV Summary if selected
            if include_csv:
                csv_buf = io.StringIO()
                csv_writer = csv.writer(csv_buf)
                csv_writer.writerow([
                    "Annotation_ID", "Chunk_ID", "File_Name", "IQ_File_Name", "Drone_Name",
                    "Category_ID", "Category_Name", "Type_of_Signal", "Protocol",
                    "TOA_us", "TOD_us", "PW_us", "FC_MHz", "BW_MHz",
                    "Freq_Low_MHz", "Freq_High_MHz", "SNR_dB",
                    "BBox_X", "BBox_Y", "BBox_W", "BBox_H", "Data_Source"
                ])

                for ann in coco_json_dict["annotations"]:
                    bdw = ann.get("bdw", {})
                    bbox = ann.get("bbox", [0, 0, 0, 0])
                    matching_img = next((img for img in coco_json_dict["images"] if img["id"] == ann["image_id"]), None)
                    fname = matching_img["file_name"] if matching_img else f"chunk_{ann['image_id']-1}.{clean_ext}"
                    iq_fname = matching_img.get("iq_file_name", "") if matching_img else ""
                    matching_cat = next((c for c in coco_json_dict["categories"] if c["id"] == ann["category_id"]), {})

                    csv_writer.writerow([
                        ann["id"],
                        ann["image_id"] - 1,
                        fname if include_images else "",
                        iq_fname if include_iq else "",
                        effective_drone,
                        ann["category_id"],
                        matching_cat.get("name", "Unknown"),
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
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        bdw.get("data_source", "")
                    ])

                zip_file.writestr("signal_parameters_bdw.csv", csv_buf.getvalue())

            total_chunks = len(rf_processor.chunks_meta)

            # 4. Render spectrogram images in parallel (if selected)
            rendered_images = {}
            if include_images and total_chunks > 0:
                done_count = 0
                lock = threading.Lock()

                def _render_task(c):
                    c_id = c["id"]
                    img_bytes, _ = rf_processor.render_spectrogram_image(
                        chunk_id=c_id,
                        width=img_width,
                        height=img_height,
                        image_format=clean_ext,
                        target_snr_db=export_snr_db
                    )
                    return c_id, img_bytes

                max_workers = min(os.cpu_count() or 4, 8)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_render_task, chunk) for chunk in rf_processor.chunks_meta]
                    for f in concurrent.futures.as_completed(futures):
                        c_id, img_bytes = f.result()
                        rendered_images[c_id] = img_bytes
                        with lock:
                            done_count += 1
                            pct = 5 + int(55.0 * (done_count / max(1, total_chunks)))
                            if progress_callback:
                                snr_txt = f" | SNR: {export_snr_db:.1f} dB" if export_snr_db is not None else ""
                                progress_callback(
                                    pct,
                                    "images",
                                    f"Rendering spectrogram image {done_count} / {total_chunks}",
                                    f"{img_width}x{img_height} {clean_ext.upper()}{snr_txt}"
                                )

                # Write rendered images into ZIP
                for chunk in rf_processor.chunks_meta:
                    c_id = chunk["id"]
                    img_name = rf_processor.get_formatted_filename(chunk_id=c_id, extension=clean_ext, drone_name=effective_drone)
                    img_bytes = rendered_images.get(c_id)
                    if img_bytes is None:
                        img_bytes, _ = rf_processor.render_spectrogram_image(
                            chunk_id=c_id,
                            width=img_width,
                            height=img_height,
                            image_format=clean_ext,
                            target_snr_db=export_snr_db
                        )
                    zip_file.writestr(f"spectrograms/{img_name}", img_bytes)

            # 5. (Optional) Add IQ chunks (Optimized: single-pass AWGN & ZIP_STORED zero-compression)
            if include_iq and rf_processor.iq_data is not None:
                global_noisy_iq = None
                if export_snr_db is not None:
                    global_noisy_iq = add_awgn(rf_processor.iq_data, snr_db=float(export_snr_db))

                for i, chunk in enumerate(rf_processor.chunks_meta):
                    c_id = chunk["id"]
                    meta = rf_processor.chunks_meta[c_id]
                    if global_noisy_iq is not None:
                        iq_slice = global_noisy_iq[meta["start_idx"]:meta["end_idx"]]
                    else:
                        iq_slice = rf_processor.iq_data[meta["start_idx"]:meta["end_idx"]]

                    iq_name = rf_processor.get_formatted_filename(chunk_id=c_id, extension="iq", drone_name=effective_drone)
                    zip_file.writestr(f"iq/{iq_name}", iq_slice.tobytes(), compress_type=zipfile.ZIP_STORED)
                    if progress_callback:
                        pct = 75 + int(10.0 * ((i + 1) / max(1, total_chunks)))
                        progress_callback(
                            pct,
                            "iq",
                            f"Slicing raw IQ chunk {i+1} / {total_chunks}",
                            f"Raw Complex64 ({len(iq_slice):,} samples)"
                        )

            # 6. (Optional) Add continuous waterfall video
            if include_video and len(rf_processor.chunks_meta) > 0:
                if progress_callback:
                    progress_callback(85, "video", "Rendering continuous waterfall MP4 video...", "Encoding H.264 frames")
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
                        tmp_vid_path = tmp_vid.name
                    rf_processor.render_waterfall_video(
                        output_filepath=tmp_vid_path,
                        fps=10.0,
                        width=img_width,
                        height=img_height
                    )
                    vid_name = f"waterfall_{effective_drone}.mp4"
                    with open(tmp_vid_path, "rb") as vf:
                        zip_file.writestr(f"video/{vid_name}", vf.read())
                    if os.path.exists(tmp_vid_path):
                        os.remove(tmp_vid_path)
                except Exception as e:
                    logger.error(f"Error embedding waterfall video in COCO ZIP: {e}")

            if progress_callback:
                progress_callback(95, "packaging", "Compressing dataset ZIP archive...", "Finalizing package")

        zip_bytes = zip_buffer.getvalue()
        logger.info(f"Generated COCO dataset ZIP bundle ({len(zip_bytes)/(1024*1024):.2f} MB, include_iq={include_iq}, include_video={include_video}, export_snr_db={export_snr_db}, drone={effective_drone}, res={img_width}x{img_height}).")
        return zip_bytes

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

        # 2. Build Image ID to Chunk Index and dimensions map
        img_id_to_chunk = {}
        img_dims = {}
        total_chunks = len(rf_processor.chunks_meta)

        for img in coco_data.get("images", []):
            img_id = img.get("id")
            img_dims[img_id] = (float(img.get("width", 1024)), float(img.get("height", 512)))

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
        target_w = float(rf_processor.render_width or img_width or 1024)
        target_h = float(rf_processor.render_height or img_height or 512)

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

            raw_w, raw_h = img_dims.get(img_id, (1024.0, 512.0))
            scale_x = target_w / raw_w if raw_w > 0 else 1.0
            scale_y = target_h / raw_h if raw_h > 0 else 1.0

            x = round(x * scale_x, 2)
            y = round(y * scale_y, 2)
            w = round(w * scale_x, 2)
            h = round(h * scale_y, 2)

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
                    img_width=int(target_w),
                    img_height=int(target_h),
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
                "img_width": int(target_w),
                "img_height": int(target_h),
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
