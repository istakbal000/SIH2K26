document.addEventListener('DOMContentLoaded', async () => {
    // 1. Fetch Config
    let mapKey = '';
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        mapKey = data.map_key;
    } catch (e) {
        console.warn("Could not fetch keys, falling back to defaults.");
    }

    // 2. Initialize Map (Restricted to India)
    const map = L.map('map', {
        center: [22.0, 79.0],
        zoom: 5,
        minZoom: 4,
        maxBounds: [
            [6.0, 68.0], // South West
            [36.0, 98.0] // North East
        ],
        maxBoundsViscosity: 1.0
    });

    // Use Esri World Street Map (similar to original Streamlit) or Mapbox if key is present
    let tileUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}';
    let attribution = 'Tiles &copy; Esri';

    if (mapKey && mapKey.length > 5) {
        // If it looks like a mapbox token
        if (mapKey.startsWith('pk.')) {
            tileUrl = `https://api.mapbox.com/styles/v1/mapbox/dark-v10/tiles/256/{z}/{x}/{y}@2x?access_token=${mapKey}`;
            attribution = 'Map data &copy; <a href="https://www.mapbox.com/">Mapbox</a>';
        }
    }

    L.tileLayer(tileUrl, {
        attribution: attribution,
        noWrap: true
    }).addTo(map);

    // Layer group for historical detections
    const historyLayer = L.layerGroup().addTo(map);
    loadHistory(historyLayer);

    // 3. Add NASA FIRMS WMS Layer
    if (mapKey && mapKey.length > 5 && !mapKey.startsWith('pk.')) {
        // VIIRS 24h gives near real-time thermal anomalies
        const firmsWmsUrl = `https://firms.modaps.eosdis.nasa.gov/mapserver/wms/fires/${mapKey}/`;
        
        const firmsLayer = L.tileLayer.wms(firmsWmsUrl, {
            layers: 'fires_viirs_24', 
            format: 'image/png',
            transparent: true,
            attribution: 'NASA FIRMS'
        });
        
        firmsLayer.addTo(map);
    } else {
        console.warn("NASA FIRMS API Key not provided (or it is a Mapbox key) - Skipping fire overlay.");
    }

    // 4. Handle Interaction
    // Allow clicking anywhere on the map to trigger analysis
    map.on('click', (e) => {
        handleMapClick(e.latlng.lat, e.latlng.lng);
    });

    async function handleMapClick(lat, lon) {
        // Update UI state
        document.getElementById('initial-state').classList.add('hidden');
        document.getElementById('results-state').classList.add('hidden');
        document.getElementById('loading-state').classList.remove('hidden');

        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lat, lon })
            });
            const data = await res.json();
            
            if (data.status === 'success') {
                updateResultsUI(lat, lon, data.results);
                
                // Update DB Status badge
                const dbBadge = document.getElementById('db-status-badge');
                if (data.database && data.database.saved) {
                    dbBadge.innerText = `💾 Saved to PostGIS (Record #${data.database.record_id} | SRID: 4326)`;
                    dbBadge.className = 'db-badge';
                    // Plot on history layer (optimistic UI)
                    loadHistory(historyLayer); 
                } else {
                    dbBadge.innerText = `⚠️ Failed to save to PostGIS`;
                    dbBadge.className = 'db-badge error';
                }
                dbBadge.classList.remove('hidden');
            }
        } catch (e) {
            console.error(e);
            alert("Error communicating with backend.");
        } finally {
            document.getElementById('loading-state').classList.add('hidden');
            document.getElementById('results-state').classList.remove('hidden');
        }
    }

    function updateResultsUI(lat, lon, results) {
        // Location
        document.getElementById('location-coords').innerHTML = `LAT: ${lat.toFixed(5)}<br>LON: ${lon.toFixed(5)}`;
        
        // Card styling based on classification
        const alertCard = document.getElementById('alert-card');
        const classEl = document.getElementById('classification-result');
        const confEl = document.getElementById('confidence-result');
        
        classEl.innerText = results.classification;
        confEl.innerText = `${results.confidence}% Confidence`;

        if (results.classification.includes("INDUSTRIAL")) {
            alertCard.classList.add('industrial');
            classEl.style.color = "var(--accent-orange)";
        } else {
            alertCard.classList.remove('industrial');
            classEl.style.color = "var(--accent-red)";
        }

        // Env values
        document.getElementById('val-temp').innerText = `${results.environmental.temperature} °C`;
        document.getElementById('val-hum').innerText = `${results.environmental.humidity} %`;
        document.getElementById('val-co2').innerText = `${results.environmental.co2} ppm`;
        document.getElementById('val-pm').innerText = `${results.environmental.pm} µg/m³`;
    }

    async function loadHistory(layerGroup) {
        try {
            const res = await fetch('/api/detections');
            const data = await res.json();
            if (data.type === 'FeatureCollection' && data.features) {
                data.features.forEach(feature => {
                    if (feature.geometry && feature.geometry.coordinates) {
                        const [lon, lat] = feature.geometry.coordinates;
                        const props = feature.properties;
                        
                        const color = props.classification.includes("INDUSTRIAL") ? "orange" : "red";
                        
                        const marker = L.circleMarker([lat, lon], {
                            radius: 8,
                            fillColor: color,
                            color: "#fff",
                            weight: 1,
                            opacity: 1,
                            fillOpacity: 0.8
                        });
                        
                        marker.bindPopup(`
                            <b>${props.classification}</b><br>
                            Conf: ${props.confidence}%<br>
                            Date: ${new Date(props.created_at).toLocaleString()}<br>
                            Temp: ${props.temperature}°C, PM: ${props.pm}µg/m³
                        `);
                        
                        layerGroup.addLayer(marker);
                    }
                });
            }
        } catch (e) {
            console.warn("Could not load history:", e);
        }
    }
});
