import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os
from collections import defaultdict
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ConstellationIdentifier:
    def __init__(self):
        self.star_catalog = self.load_star_data()
        self.constellation_data = self.load_constellation_data()
        print(f"Loaded {len(self.star_catalog)} stars and {len(self.constellation_data)} constellations")

    def load_star_data(self):
        """Load star data with comprehensive constellations"""
        data = {
            'proper': [
                # Orion
                'Betelgeuse', 'Rigel', 'Bellatrix', 'Mintaka', 'Alnilam', 'Alnitak', 'Saiph',
                # Canis Major
                'Sirius', 'Adhara', 'Wezen', 'Mirzam', 'Aludra',
                # Gemini
                'Castor', 'Pollux', 'Alhena', 'Wasat', 'Mebsuta',
                # Cassiopeia
                'Schedar', 'Caph', 'Gamma Cassiopeiae', 'Ruchbah', 'Segin',
                # Cygnus
                'Deneb', 'Albireo', 'Sadr', 'Gienah', 'Delta Cygni',
                # Lyra
                'Vega', 'Sheliak', 'Sulafat', 'Delta Lyrae', 'Zeta Lyrae',
                # Scorpius
                'Antares', 'Shaula', 'Sargas', 'Dschubba', 'Acrab',
                # Leo
                'Regulus', 'Denebola', 'Algieba', 'Zosma', 'Adhafera',
                # Ursa Major
                'Dubhe', 'Merak', 'Phecda', 'Megrez', 'Alioth', 'Mizar', 'Alkaid'
            ],
            'ra': [
                # Orion
                88.792939, 78.634467, 81.282764, 83.001667, 84.053389, 85.189703, 86.939119,
                # Canis Major
                101.287155, 104.656453, 107.097850, 95.674939, 111.023763,
                # Gemini
                113.649428, 116.328958, 99.427961, 110.030739, 100.983140,
                # Cassiopeia
                10.126838, 2.294522, 14.177052, 21.453806, 28.598895,
                # Cygnus
                310.357980, 292.680175, 305.557091, 304.513871, 299.074096,
                # Lyra
                279.234735, 282.519978, 283.633359, 284.735929, 285.697806,
                # Scorpius
                247.351915, 263.402167, 255.986760, 240.083359, 241.359301,
                # Leo
                152.092962, 177.264910, 154.993144, 168.520019, 154.172567,
                # Ursa Major
                165.931965, 165.460319, 168.527018, 168.560017, 193.507290, 200.981429, 206.885157
            ],
            'dec': [
                # Orion
                7.407064, -8.201638, 6.349703, -0.299095, -1.201919, -1.942574, -9.669605,
                # Canis Major
                -16.716116, -28.972086, -26.393200, -17.955919, -29.303100,
                # Gemini
                31.888276, 28.026183, 16.399281, 21.982316, 25.131100,
                # Cassiopeia
                56.537331, 59.149781, 60.716828, 60.235284, 63.670101,
                # Cygnus
                45.280339, 27.959691, 40.256679, 33.970486, 38.922296,
                # Lyra
                38.783689, 33.362667, 32.689557, 36.898611, 37.605114,
                # Scorpius
                -26.432003, -37.103824, -42.997824, -22.621710, -19.805453,
                # Leo
                11.967209, 14.572058, 19.841489, 20.523718, 23.417312,
                # Ursa Major
                61.751035, 56.382426, 53.694759, 57.032615, 55.959823, 54.925362, 49.313267
            ],
            'mag': [
                # Orion
                0.45, 0.18, 1.64, 2.25, 1.69, 1.74, 2.07,
                # Canis Major
                -1.46, 1.50, 1.83, 1.98, 2.45,
                # Gemini
                1.58, 1.16, 1.93, 3.53, 3.06,
                # Cassiopeia
                2.24, 2.28, 2.15, 2.66, 3.35,
                # Cygnus
                1.25, 3.05, 2.23, 2.48, 2.87,
                # Lyra
                0.03, 3.52, 3.25, 4.22, 4.34,
                # Scorpius
                1.06, 1.62, 1.86, 2.29, 2.56,
                # Leo
                1.36, 2.14, 2.61, 2.56, 3.44,
                # Ursa Major
                1.81, 2.37, 2.44, 3.32, 1.76, 2.23, 1.85
            ],
            'con': [
                # Orion
                'Ori', 'Ori', 'Ori', 'Ori', 'Ori', 'Ori', 'Ori',
                # Canis Major
                'CMa', 'CMa', 'CMa', 'CMa', 'CMa',
                # Gemini
                'Gem', 'Gem', 'Gem', 'Gem', 'Gem',
                # Cassiopeia
                'Cas', 'Cas', 'Cas', 'Cas', 'Cas',
                # Cygnus
                'Cyg', 'Cyg', 'Cyg', 'Cyg', 'Cyg',
                # Lyra
                'Lyr', 'Lyr', 'Lyr', 'Lyr', 'Lyr',
                # Scorpius
                'Sco', 'Sco', 'Sco', 'Sco', 'Sco',
                # Leo
                'Leo', 'Leo', 'Leo', 'Leo', 'Leo',
                # Ursa Major
                'UMa', 'UMa', 'UMa', 'UMa', 'UMa', 'UMa', 'UMa'
            ]
        }
        return pd.DataFrame(data).to_dict('records')

    def load_constellation_data(self):
        """Load constellation patterns for all requested constellations"""
        return {
            'Orion': {
                'stars': ['Betelgeuse', 'Rigel', 'Bellatrix', 'Mintaka', 'Alnilam', 'Alnitak', 'Saiph'],
                'lines': [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (1, 0)],
                'bounds': {'ra': (70, 100), 'dec': (-20, 20)},
                'min_stars': 4
            },
            'Canis Major': {
                'stars': ['Sirius', 'Adhara', 'Wezen', 'Mirzam', 'Aludra'],
                'lines': [(0, 1), (1, 2), (0, 3), (3, 4)],
                'bounds': {'ra': (95, 115), 'dec': (-30, -15)},
                'min_stars': 3
            },
            'Gemini': {
                'stars': ['Castor', 'Pollux', 'Alhena', 'Wasat', 'Mebsuta'],
                'lines': [(0, 1), (1, 2), (2, 3), (3, 4)],
                'bounds': {'ra': (95, 120), 'dec': (10, 35)},
                'min_stars': 3
            },
            'Cassiopeia': {
                'stars': ['Schedar', 'Caph', 'Gamma Cassiopeiae', 'Ruchbah', 'Segin'],
                'lines': [(0, 1), (1, 2), (2, 3), (3, 4)],
                'bounds': {'ra': (0, 30), 'dec': (50, 70)},
                'min_stars': 3
            },
            'Cygnus': {
                'stars': ['Deneb', 'Albireo', 'Sadr', 'Gienah', 'Delta Cygni'],
                'lines': [(0, 2), (2, 3), (3, 1), (2, 4)],
                'bounds': {'ra': (290, 315), 'dec': (25, 50)},
                'min_stars': 3
            },
            'Lyra': {
                'stars': ['Vega', 'Sheliak', 'Sulafat', 'Delta Lyrae', 'Zeta Lyrae'],
                'lines': [(0, 1), (0, 2), (1, 3), (2, 4)],
                'bounds': {'ra': (275, 290), 'dec': (30, 40)},
                'min_stars': 3
            },
            'Scorpius': {
                'stars': ['Antares', 'Shaula', 'Sargas', 'Dschubba', 'Acrab'],
                'lines': [(0, 3), (3, 4), (4, 2), (2, 1)],
                'bounds': {'ra': (235, 265), 'dec': (-45, -20)},
                'min_stars': 3
            },
            'Leo': {
                'stars': ['Regulus', 'Denebola', 'Algieba', 'Zosma', 'Adhafera'],
                'lines': [(0, 2), (2, 3), (3, 1), (2, 4)],
                'bounds': {'ra': (140, 180), 'dec': (0, 30)},
                'min_stars': 3
            },
            'Ursa Major': {
                'stars': ['Dubhe', 'Merak', 'Phecda', 'Megrez', 'Alioth', 'Mizar', 'Alkaid'],
                'lines': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
                'bounds': {'ra': (160, 220), 'dec': (40, 70)},
                'min_stars': 4
            }
        }

    def detect_stars(self, image_path):
        """Improved star detection with adaptive parameters"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Could not load image")
            
            # Adaptive preprocessing
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
            
            # Morphological cleanup
            kernel = np.ones((3,3), np.uint8)
            cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            
            # Find stars
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            stars = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2 <= area <= 50:  # Adjusted star size range
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        x = int(M["m10"] / M["m00"])
                        y = int(M["m01"] / M["m00"])
                        brightness = np.mean(img[max(0,y-2):min(img.shape[0],y+2), 
                                           max(0,x-2):min(img.shape[1],x+2)])
                        if brightness > 30:  # Minimum brightness threshold
                            stars.append({
                                'x': x, 
                                'y': y, 
                                'brightness': brightness,
                                'estimated_mag': 6 - (brightness/40)  # Simple magnitude estimation
                            })
            
            print(f"Detected {len(stars)} stars")
            return img, stars
        
        except Exception as e:
            print(f"Star detection error: {e}")
            return None, []

    def match_constellation(self, stars, img_size):
        """Advanced constellation matching with multiple checks"""
        if len(stars) < 3:
            return None, 0
        
        img_w, img_h = img_size
        
        # Convert image coords to celestial coords (simplified)
        star_coords = []
        for star in stars:
            ra = (star['x'] / img_w) * 360
            dec = ((star['y'] / img_h) - 0.5) * 180
            star_coords.append({
                'ra': ra,
                'dec': dec,
                'mag': star['estimated_mag']
            })
        
        # Prepare catalog data
        catalog_coords = np.array([[s['ra'], s['dec'], s['mag']] for s in self.star_catalog])
        star_names = [s['proper'] for s in self.star_catalog]
        const_abbr = [s['con'] for s in self.star_catalog]
        
        # Constellation abbreviation mapping
        const_map = {
            'Ori': 'Orion',
            'CMa': 'Canis Major',
            'Gem': 'Gemini',
            'Cas': 'Cassiopeia',
            'Cyg': 'Cygnus',
            'Lyr': 'Lyra',
            'Sco': 'Scorpius',
            'Leo': 'Leo',
            'UMa': 'Ursa Major'
        }
        
        try:
            # Find nearest catalog stars (using RA/Dec and magnitude)
            nn = NearestNeighbors(n_neighbors=1, 
                                metric=lambda a, b: np.sqrt(
                                    ((a[0]-b[0])**2 + (a[1]-b[1])**2) + 0.5*(abs(a[2]-b[2]))))
            nn.fit(catalog_coords)
            
            detected = np.array([[s['ra'], s['dec'], s['mag']] for s in star_coords])
            distances, indices = nn.kneighbors(detected)
            
            # Count matches per constellation
            const_scores = defaultdict(int)
            matched_stars = []
            
            for i, idx in enumerate(indices.flatten()):
                const = const_abbr[idx]
                if const and const != 'nan':
                    const_scores[const] += 1
                    matched_stars.append({
                        'detected': star_coords[i],
                        'catalog': self.star_catalog[idx]
                    })
            
            # Verify matches against constellation requirements
            valid_constellations = {}
            for abbr, count in const_scores.items():
                full_name = const_map.get(abbr, abbr)
                if full_name in self.constellation_data:
                    const_data = self.constellation_data[full_name]
                    min_stars = const_data['min_stars']
                    
                    if count >= min_stars:
                        # Check if matches are within constellation bounds
                        in_bounds = 0
                        bounds = const_data['bounds']
                        for match in matched_stars:
                            if (match['catalog']['con'] == abbr and
                                bounds['ra'][0] <= match['detected']['ra'] <= bounds['ra'][1] and
                                bounds['dec'][0] <= match['detected']['dec'] <= bounds['dec'][1]):
                                in_bounds += 1
                        
                        # Calculate confidence score
                        if in_bounds >= min_stars:
                            total_stars = len(const_data['stars'])
                            confidence = min(0.99, (count / total_stars) * (1 + in_bounds/total_stars))
                            valid_constellations[full_name] = confidence
            
            if not valid_constellations:
                return None, 0
            
            # Return best match
            best_match = max(valid_constellations.items(), key=lambda x: x[1])
            return best_match[0], best_match[1]
            
        except Exception as e:
            print(f"Matching error: {e}")
            return None, 0

    def visualize(self, image_path, stars, constellation):
        """Enhanced visualization with constellation lines"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image for visualization")
            
            # Draw all detected stars (size by brightness)
            for star in stars:
                size = int(5 * (1 - star['estimated_mag']/6))  # Brighter = larger
                cv2.circle(img, (star['x'], star['y']), max(2, size), (0, 255, 0), -1)
            
            # Draw constellation if identified
            if constellation and constellation in self.constellation_data:
                const = self.constellation_data[constellation]
                
                # Get positions of constellation stars in image
                star_positions = {}
                for i, star_name in enumerate(const['stars']):
                    for catalog_star in self.star_catalog:
                        if catalog_star['proper'] == star_name:
                            # Convert RA/Dec to image coordinates
                            x = int((catalog_star['ra'] / 360) * img.shape[1])
                            y = int(((catalog_star['dec'] / 180) + 0.5) * img.shape[0])
                            star_positions[i] = (x, y)
                            # Label bright stars
                            if catalog_star['mag'] < 2.0:
                                cv2.putText(img, star_name, (x+10, y-5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                            break
                
                # Draw constellation lines
                for i, j in const['lines']:
                    if i in star_positions and j in star_positions:
                        cv2.line(img, star_positions[i], star_positions[j], (0, 0, 255), 2)
                
                # Add constellation name
                cv2.putText(img, constellation, (30, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display
            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization error: {e}")

def main():
    print("\n=== Advanced Constellation Identifier ===")
    print("Supports: Orion, Canis Major, Gemini, Cassiopeia, Cygnus, Lyra, Scorpius, Leo, Ursa Major\n")
    
    # Get valid image path
    while True:
        image_path = input("Enter path to your night sky image: ").strip()
        if os.path.exists(image_path):
            break
        print("File not found. Please try again.")
    
    try:
        # Initialize and process
        identifier = ConstellationIdentifier()
        
        print("\nProcessing image...")
        img, stars = identifier.detect_stars(image_path)
        
        if len(stars) < 3:
            print("\nNot enough stars detected. Try a clearer/darker image.")
            return
        
        print("Matching constellations...")
        constellation, confidence = identifier.match_constellation(stars, (img.shape[1], img.shape[0]))
        
        # Show results
        if constellation:
            print(f"\nIdentified: {constellation} (confidence: {confidence:.0%})")
            identifier.visualize(image_path, stars, constellation)
        else:
            print("\nNo confident match found. Possible reasons:")
            print("- Constellation not in database")
            print("- Too few stars visible")
            print("- Image quality issues")
            identifier.visualize(image_path, stars, None)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your image and try again.")

if __name__ == "__main__":
    main()