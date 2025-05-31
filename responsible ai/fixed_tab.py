# This is a self-contained fix for the Individual Analysis tab
# Add this code to the app.py file to replace the problematic tab

# Tab 3: Individual detection analysis with enhanced visualization
with tab3:
    # Add clear header for Individual Heatmaps
    st.subheader("Individual Heatmaps for Each Detection")
    
    if len(filtered_boxes) > 0:
        # Add a card with explanation
        st.markdown("""
        <div class="card">
            <div class="card-title">Individual Heatmap Analysis</div>
            <p>Interactive analysis of each detected palm tree with detailed Grad-CAM visualization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a progress bar for heatmap generation
        progress_bar = st.progress(0)
        st.info("Generating individual heatmaps... Please wait.")
        
        # Pre-compute all heatmaps with progress updates
        heatmaps = []
        with st.spinner("Processing heatmaps..."):
            for idx, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                # Update progress bar
                progress = (idx + 1) / len(filtered_boxes)
                progress_bar.progress(progress)
                
                # Generate heatmap for this detection
                x1, y1, x2, y2 = box
                crop = image.crop((x1, y1, x2, y2))
                gradcam_crop = generate_gradcam_visualization(_model=model, image=image, box=box, score=score)
                
                # Calculate tree dimensions in pixels and relative size
                width_px = x2 - x1
                height_px = y2 - y1
                area_px = width_px * height_px
                relative_size = (area_px / (image.width * image.height)) * 100
                aspect_ratio = width_px / height_px if height_px > 0 else 0
                
                # Store for display
                heatmaps.append({
                    'idx': idx,
                    'box': box,
                    'score': score,
                    'crop': crop,
                    'gradcam': gradcam_crop,
                    'width': width_px,
                    'height': height_px,
                    'area': area_px,
                    'relative_size': relative_size,
                    'aspect_ratio': aspect_ratio
                })
        
        # Clear the progress indicators
        progress_bar.empty()
        st.success("All heatmaps generated successfully!")
        
        # Add a selection method with a UNIQUE KEY
        display_option = st.radio(
            "Choose display method:",
            ["Grid View", "Detailed Individual View"],
            horizontal=True,
            key="display_option_unique_key_1"  # Add a unique key here
        )
        
        if display_option == "Grid View":
            # Create a grid of columns for displaying individual heatmaps
            cols_per_row = min(3, len(heatmaps))  # Max 3 columns per row
            for i in range(0, len(heatmaps), cols_per_row):
                # Create columns for this row
                cols = st.columns(cols_per_row)
                
                # For each column in this row
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(heatmaps):
                        with cols[j]:
                            heatmap_data = heatmaps[idx]
                            score = heatmap_data['score']
                            
                            # Add a card-like container for each detection
                            st.markdown(f"""
                            <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                                <div style="font-weight: 600; color: #1E88E5; margin-bottom: 8px; font-size: 1.1rem;">Detection #{idx+1}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 1. First show the original crop with box
                            st.image(heatmap_data['crop'], caption=f"Original Detection", use_container_width=True)
                            
                            # 2. Then show the GradCAM heatmap
                            st.image(heatmap_data['gradcam'], caption=f"GradCAM Heatmap", use_container_width=True)
                            
                            # 3. Show key metrics
                            st.markdown(f"""
                            <div style="background-color: rgba(30, 136, 229, 0.05); padding: 10px; border-radius: 5px; margin-top: 10px;">
                                <div style="color: #666; font-size: 0.9rem;"><strong>Confidence:</strong> {score:.2f}</div>
                                <div style="color: #666; font-size: 0.9rem;"><strong>Size:</strong> {heatmap_data['width']}×{heatmap_data['height']} px</div>
                                <div style="color: #666; font-size: 0.9rem;"><strong>Area:</strong> {heatmap_data['area']} px²</div>
                                <div style="color: #666; font-size: 0.9rem;"><strong>Relative Size:</strong> {heatmap_data['relative_size']:.1f}% of image</div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            # Select which detection to view in detail
            detection_options = [f"Detection #{i+1} (Confidence: {h['score']:.2f})" for i, h in enumerate(heatmaps)]
            selected_detection = st.selectbox(
                "Select a detection to analyze in detail:", 
                detection_options, 
                key="detailed_view_selectbox"  # Add a unique key
            )
            selected_idx = detection_options.index(selected_detection)
            
            # Get the selected heatmap data
            heatmap_data = heatmaps[selected_idx]
            
            # Show detailed view with side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">Original Detection</div>
                    <p>Palm tree as identified by the detection model</p>
                </div>
                """, unsafe_allow_html=True)
                st.image(heatmap_data['crop'], use_container_width=True)
                
                # Show statistics for this detection
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-header">Detection Statistics</div>
                    <div class="result-content">
                        <table style="width:100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Confidence Score:</strong></td>
                                <td style="padding: 8px; border-bottom: 1px solid #eee;">{heatmap_data['score']:.4f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Dimensions:</strong></td>
                                <td style="padding: 8px; border-bottom: 1px solid #eee;">{heatmap_data['width']}×{heatmap_data['height']} pixels</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Detection Area:</strong></td>
                                <td style="padding: 8px; border-bottom: 1px solid #eee;">{heatmap_data['area']} px²</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Aspect Ratio:</strong></td>
                                <td style="padding: 8px; border-bottom: 1px solid #eee;">{heatmap_data['aspect_ratio']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px;"><strong>Relative Size:</strong></td>
                                <td style="padding: 8px;">{heatmap_data['relative_size']:.2f}% of image</td>
                            </tr>
                        </table>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">Grad-CAM Heatmap</div>
                    <p>Visual explanation of model's detection decision</p>
                </div>
                """, unsafe_allow_html=True)
                st.image(heatmap_data['gradcam'], use_container_width=True)
                
                # Explanation of the heatmap
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-header">Heatmap Interpretation</div>
                    <div class="result-content">
                        <p>The Grad-CAM heatmap visualizes which parts of the image most influenced the model's decision:</p>
                        <ul>
                            <li><strong>Red/Yellow areas:</strong> Features strongly contributing to palm tree detection</li>
                            <li><strong>Blue/Green areas:</strong> Features with less contribution to the detection</li>
                        </ul>
                        <p>This helps understand what visual patterns the model recognizes as characteristics of palm trees.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No detections to analyze. Please upload an image with palm trees.")
