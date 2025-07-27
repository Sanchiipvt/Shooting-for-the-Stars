document.addEventListener('DOMContentLoaded', function() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('skyUpload');
  const resultsContainer = document.getElementById('resultsContainer');
  const retryBtn = document.getElementById('retryBtn');
  const canvas = document.getElementById('starCanvas');
  const constellationOverlay = document.getElementById('constellationOverlay');
  const constellationList = document.getElementById('constellationList');
  
  // Sample constellation data (without star details)
  const sampleConstellations = [
    { name: "Orion", confidence: 92 },
    { name: "Ursa Major", confidence: 87 },
    { name: "Cassiopeia", confidence: 85 },
    { name: "Leo", confidence: 78 }
  ];

  // Handle drag and drop
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
  });

  function highlight() {
    dropZone.classList.add('active');
  }

  function unhighlight() {
    dropZone.classList.remove('active');
  }

  dropZone.addEventListener('drop', handleDrop, false);
  fileInput.addEventListener('change', handleFiles, false);
  dropZone.addEventListener('click', () => fileInput.click());

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files } });
  }

  function handleFiles(e) {
    const files = e.target.files;
    if (files.length) {
      const file = files[0];
      if (file.type.match('image.*')) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
          displayImage(e.target.result);
          processImage(e.target.result);
        };
        
        reader.readAsDataURL(file);
      }
    }
  }

  function displayImage(imageData) {
    resultsContainer.style.display = 'block';
    retryBtn.style.display = 'block';
    
    setTimeout(() => {
      resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }, 100);
    
    const img = new Image();
    img.onload = function() {
      canvas.width = img.width;
      canvas.height = img.height;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      
      setTimeout(() => {
        drawConstellationData();
      }, 1500);
    };
    img.src = imageData;
  }

  function drawConstellationData() {
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = function() {
      ctx.drawImage(img, 0, 0);
      drawStars(ctx);
      drawConstellationLines(ctx);
      populateConstellationInfo();
    };
    img.src = canvas.toDataURL();
  }

  function drawStars(ctx) {
    const starCount = 50 + Math.floor(Math.random() * 50);
    
    for (let i = 0; i < starCount; i++) {
      const x = Math.random() * canvas.width;
      const y = Math.random() * canvas.height;
      const size = 1 + Math.random() * 3;
      
      const gradient = ctx.createRadialGradient(
        x, y, 0,
        x, y, size * 3
      );
      gradient.addColorStop(0, 'rgba(255, 215, 0, 0.8)');
      gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');
      
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI);
      ctx.fillStyle = gradient;
      ctx.fill();
      
      ctx.beginPath();
      ctx.arc(x, y, size/2, 0, 2 * Math.PI);
      ctx.fillStyle = '#FFD700';
      ctx.fill();
    }
  }

  function drawConstellationLines(ctx) {
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.7)';
    ctx.lineWidth = 2;
    
    const lineCount = 10 + Math.floor(Math.random() * 10);
    
    for (let i = 0; i < lineCount; i++) {
      const x1 = Math.random() * canvas.width;
      const y1 = Math.random() * canvas.height;
      const x2 = Math.random() * canvas.width;
      const y2 = Math.random() * canvas.height;
      
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      
      ctx.shadowBlur = 10;
      ctx.shadowColor = 'rgba(0, 229, 255, 0.5)';
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }

  function populateConstellationInfo() {
    constellationList.innerHTML = '';
    
    sampleConstellations.forEach(constellation => {
      const card = document.createElement('div');
      card.className = 'constellation-card';
      
      const name = document.createElement('h4');
      name.textContent = constellation.name;
      
      const confidence = document.createElement('div');
      confidence.className = 'confidence-badge';
      confidence.textContent = `${constellation.confidence}% match`;
      
      card.appendChild(name);
      card.appendChild(confidence);
      constellationList.appendChild(card);
    });
  }

  retryBtn.addEventListener('click', function() {
    resultsContainer.style.display = 'none';
    retryBtn.style.display = 'none';
    fileInput.value = '';
    document.querySelector('.hero').scrollIntoView({ behavior: 'smooth' });
  });
});