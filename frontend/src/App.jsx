import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [ingredients, setIngredients] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [missing, setMissing] = useState('');
  const [adaptedStep, setAdaptedStep] = useState('');
  const [adaptLoading, setAdaptLoading] = useState(false);
  const [videos, setVideos] = useState([]);
  const [videosLoading, setVideosLoading] = useState(false);

  // Image-based ingredient detection
  // Option 1: Single combined image
  const [singleImageFile, setSingleImageFile] = useState(null);
  const [singleImagePreview, setSingleImagePreview] = useState(null);
  const [detectingSingle, setDetectingSingle] = useState(false);
  // Option 2: Multiple individual images
  const [multiImageFiles, setMultiImageFiles] = useState([]);
  const [multiImagePreviews, setMultiImagePreviews] = useState([]);
  const [detectingMulti, setDetectingMulti] = useState(false);
  const [cnnDetected, setCnnDetected] = useState([]);
  const [llmDetected, setLlmDetected] = useState([]);

  // Refs for file inputs to reset them
  const singleImageInputRef = useRef(null);
  const multiImageInputRef = useRef(null);

  // Fetch recipes from Flask backend
  const searchRecipes = async () => {
    if (!ingredients.trim()) return;
    
    setLoading(true);
    setError('');
    try {
      const response = await fetch(
        `http://127.0.0.1:5000/search?q=${encodeURIComponent(ingredients)}`
      );
      if (!response.ok) throw new Error('Failed to fetch recipes');
      const data = await response.json();
      setResults(data);
      if (data.length === 0) {
        setError('No recipes found. Try different ingredients!');
      }
    } catch (error) {
      setError('Error fetching recipes. Make sure the server is running!');
      console.error(error);
    }
    setLoading(false);
  };

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !loading && ingredients.trim()) {
      searchRecipes();
    }
  };

  // Handle single combined image upload
  const handleSingleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file');
      return;
    }

    setSingleImageFile(file);
    setSingleImagePreview(null);
    setCnnDetected([]);
    setLlmDetected([]);
    setError('');
    setDetectingSingle(false);

    const reader = new FileReader();
    reader.onloadend = () => {
      setSingleImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  // Handle multiple individual image upload (store files + previews)
  const handleMultiImageUpload = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const validImages = files.filter((file) => file.type.startsWith('image/'));
    if (validImages.length === 0) {
      setError('Please upload valid image files');
      return;
    }

    setMultiImageFiles(validImages);
    setCnnDetected([]);
    setLlmDetected([]);
    setError('');
    setDetectingMulti(false);

    // Generate previews
    const previews = [];
    validImages.forEach((file, index) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        previews[index] = reader.result;
        if (previews.length === validImages.length) {
          setMultiImagePreviews([...previews]);
        }
      };
      reader.readAsDataURL(file);
    });
  };

  // Remove single image
  const removeSingleImage = (e) => {
    e.stopPropagation();
    setSingleImageFile(null);
    setSingleImagePreview(null);
    setCnnDetected([]);
    setLlmDetected([]);
    // Reset the file input
    if (singleImageInputRef.current) {
      singleImageInputRef.current.value = '';
    }
  };

  // Remove image from multiple images
  const removeMultiImage = (index, e) => {
    e.stopPropagation();
    const newFiles = multiImageFiles.filter((_, i) => i !== index);
    const newPreviews = multiImagePreviews.filter((_, i) => i !== index);
    setMultiImageFiles(newFiles);
    setMultiImagePreviews(newPreviews);
    setCnnDetected([]);
    setLlmDetected([]);
    // Reset the file input if all images are removed
    if (newFiles.length === 0 && multiImageInputRef.current) {
      multiImageInputRef.current.value = '';
    }
  };

  // Detect ingredients from a single combined image
  const detectFromSingleImage = async () => {
    if (!singleImageFile) return;
    setDetectingSingle(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('image', singleImageFile);

      const response = await fetch('http://127.0.0.1:5000/classify-image?mode=llm_only', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to classify image');
      }

      const cnnList = Array.isArray(data.cnn_ingredients) ? data.cnn_ingredients : [];
      const llmList = Array.isArray(data.ingredients) ? data.ingredients : [];

      setCnnDetected(cnnList);
      setLlmDetected(llmList);

      if (llmList.length > 0) {
        setIngredients(llmList.join(', '));
      } else {
        setError('No ingredients detected from image');
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'Error detecting ingredients from image');
    } finally {
      setDetectingSingle(false);
    }
  };

  // Detect ingredients from multiple individual images
  const detectFromMultiImages = async () => {
    if (multiImageFiles.length === 0) return;
    setDetectingMulti(true);
    setError('');

    try {
      const formData = new FormData();
      multiImageFiles.forEach((file) => {
        formData.append('images', file);
      });

      const response = await fetch('http://127.0.0.1:5000/classify-image?mode=cnn', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to classify image');
      }

      const cnnList = Array.isArray(data.cnn_ingredients) ? data.cnn_ingredients : [];
      const llmList = Array.isArray(data.ingredients) ? data.ingredients : [];

      setCnnDetected(cnnList);
      setLlmDetected(llmList);

      if (llmList.length > 0) {
        setIngredients(llmList.join(', '));
      } else {
        setError('No ingredients detected from image');
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'Error detecting ingredients from image');
    } finally {
      setDetectingMulti(false);
    }
  };

  // Handle view/adapt modal opening
  function openRecipe(recipe) {
    setSelectedRecipe(recipe);
    setMissing('');
    setAdaptedStep('');
    setVideos([]);
    setVideosLoading(true);
  }

  // Close modal
  const closeModal = () => {
    setSelectedRecipe(null);
    setMissing('');
    setAdaptedStep('');
    setVideos([]);
  };

  // Format AI suggestion text into bullet points
  const formatAdaptationText = (text) => {
    if (!text) return [];
    
    // Split by common patterns: numbered lists, bullet points, line breaks, or periods
    // First, try to split by numbered items (1., 2., etc.)
    let items = text.split(/(?=\d+\.\s)/).filter(item => item.trim());
    
    // If no numbered items, try splitting by line breaks
    if (items.length <= 1) {
      items = text.split(/\n+/).filter(item => item.trim());
    }
    
    // If still single item, try splitting by periods followed by space
    if (items.length <= 1) {
      items = text.split(/\.\s+/).filter(item => item.trim() && item.length > 10);
    }
    
    // Clean up items: remove leading numbers/bullets, trim whitespace
    return items.map(item => {
      // Remove leading numbers, bullets, dashes
      item = item.replace(/^[\d\-\•\*\u2022]\s*/, '').trim();
      // Remove trailing period if present
      item = item.replace(/\.$/, '');
      return item;
    }).filter(item => item.length > 0);
  };

  // Call backend to get AI adaptation
  const getAdaptationAdvice = async () => {
    if (!missing.trim()) return;
    
    setAdaptLoading(true);
    setAdaptedStep('');
    try {
      const response = await fetch('http://127.0.0.1:5000/adapt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instructions: selectedRecipe.instructions,
          missing: missing,
          title: selectedRecipe.title,
        }),
      });
      if (!response.ok) throw new Error('Failed to get adaptation');
      const data = await response.json();
      setAdaptedStep(data.adaptedStep);
    } catch (err) {
      setAdaptedStep(
        'AI adaptation could not be fetched. Please check your backend and API configuration.'
      );
    }
    setAdaptLoading(false);
  };

  // Fetch YouTube videos when recipe modal opens
  useEffect(() => {
    if (selectedRecipe) {
      setVideosLoading(true);
      fetch(
        `http://127.0.0.1:5000/videos?recipe=${encodeURIComponent(
          selectedRecipe.title
        )}`
      )
        .then((response) => response.json())
        .then((data) => {
          if (Array.isArray(data)) {
            setVideos(data);
          } else {
            setVideos([]);
          }
        })
        .catch(() => setVideos([]))
        .finally(() => setVideosLoading(false));
    }
  }, [selectedRecipe]);

  return (
    <div className="app">
      <div className="app-container">
        <header className="header">
          <div className="header-inner">
            <div className="logo-section">
              <div className="logo-circle">
                <span className="logo-text">PM</span>
              </div>
              <div className="logo-content">
                <h1 className="logo-title">PantryMatch</h1>
                <p className="logo-subtitle">Cook smarter, not harder</p>
              </div>
            </div>
          </div>
        </header>

        <div className="content-wrapper">
          <div className="search-area">
            <div className="search-box">
              <div className="search-icon-wrapper">
                <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <circle cx="11" cy="11" r="8"></circle>
                  <path d="m21 21-4.35-4.35"></path>
                </svg>
              </div>
              <input
                type="text"
                className="search-field"
                placeholder="What's in your pantry? (e.g., chicken, tomatoes, rice)"
                value={ingredients}
                onChange={(e) => setIngredients(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={loading}
              />
              <button
                className="search-btn"
                onClick={searchRecipes}
                disabled={loading || !ingredients.trim()}
              >
                {loading ? (
                  <span className="btn-loader"></span>
                ) : (
                  <span>Find Recipes</span>
                )}
              </button>
            </div>

            {error && (
              <div className="alert alert-error">
                <svg className="alert-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                {error}
              </div>
            )}

            {/* Option 1: single combined image */}
            <div className="image-upload-section">
              <p className="image-upload-title">Option 1: Upload a single image with all ingredients</p>
              {singleImagePreview ? (
                <div className="image-upload-box" style={{ pointerEvents: 'none' }}>
                  <input
                    ref={singleImageInputRef}
                    type="file"
                    accept="image/*"
                    className="image-upload-input"
                    onChange={handleSingleImageUpload}
                    disabled={detectingSingle}
                  />
                  <div className="image-upload-preview-wrapper">
                    <div className="image-preview-container">
                      <img
                        src={singleImagePreview}
                        alt="Combined ingredients preview"
                        className="image-upload-preview"
                      />
                      <button
                        className="image-remove-btn"
                        onClick={removeSingleImage}
                        aria-label="Remove image"
                        style={{ pointerEvents: 'auto' }}
                      >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <line x1="18" y1="6" x2="6" y2="18"></line>
                          <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <label className="image-upload-box">
                  <input
                    ref={singleImageInputRef}
                    type="file"
                    accept="image/*"
                    className="image-upload-input"
                    onChange={handleSingleImageUpload}
                    disabled={detectingSingle}
                  />
                  <div className="image-upload-placeholder">
                    <span className="image-upload-icon">📸</span>
                    <span className="image-upload-main">Drop a single image or click to browse</span>
                    <span className="image-upload-hint">
                      Best when all ingredients are visible in one clear photo
                    </span>
                  </div>
                </label>
              )}
            </div>

            {/* Option 2: multiple individual images */}
            <div className="image-upload-section" style={{ marginTop: 24 }}>
              <p className="image-upload-title">Option 2: Upload separate images of ingredients</p>
              {multiImagePreviews && multiImagePreviews.length > 0 ? (
                <div className="image-upload-box" style={{ pointerEvents: 'none' }}>
                  <input
                    ref={multiImageInputRef}
                    type="file"
                    multiple
                    accept="image/*"
                    className="image-upload-input"
                    onChange={handleMultiImageUpload}
                    disabled={detectingMulti}
                  />
                  <div className="image-upload-preview-wrapper">
                    {multiImagePreviews.map((src, idx) => (
                      <div key={idx} className="image-preview-container">
                        <img
                          src={src}
                          alt={`Ingredient ${idx + 1}`}
                          className="image-upload-preview"
                        />
                        <button
                          className="image-remove-btn"
                          onClick={(e) => removeMultiImage(idx, e)}
                          aria-label="Remove image"
                          style={{ pointerEvents: 'auto' }}
                        >
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <label className="image-upload-box">
                  <input
                    ref={multiImageInputRef}
                    type="file"
                    multiple
                    accept="image/*"
                    className="image-upload-input"
                    onChange={handleMultiImageUpload}
                    disabled={detectingMulti}
                  />
                  <div className="image-upload-placeholder">
                    <span className="image-upload-icon">📸</span>
                    <span className="image-upload-main">Drop multiple images or click to browse</span>
                    <span className="image-upload-hint">
                      Best when each ingredient is in its own clear photo
                    </span>
                  </div>
                </label>
              )}
            </div>

            {/* Single combined detect button */}
            <div className="image-upload-section" style={{ marginTop: 16 }}>
              <button
                className="image-upload-button"
                onClick={async () => {
                  if (multiImageFiles.length > 0) {
                    await detectFromMultiImages();
                  } else if (singleImageFile) {
                    await detectFromSingleImage();
                  }
                }}
                disabled={
                  detectingSingle ||
                  detectingMulti ||
                  (!singleImageFile && multiImageFiles.length === 0)
                }
              >
                {detectingSingle || detectingMulti
                  ? 'Detecting all ingredients...'
                  : 'Detect all ingredients'}
              </button>
            </div>

            {(cnnDetected.length > 0 || llmDetected.length > 0) && (
              <div className="detected-summary">
                <div className="detected-group">
                  <p className="detected-label">Ingredients detected by ResNet model</p>
                  {cnnDetected.length > 0 ? (
                    <div className="image-model-chips">
                      {cnnDetected.map((ing, idx) => (
                        <span key={`${ing}-${idx}`} className="image-model-chip">
                          {ing}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="detected-empty">No ResNet ingredients detected yet.</p>
                  )}
                </div>

                <div className="detected-group">
                  <p className="detected-label">All ingredients available (OpenRouter)</p>
                  {llmDetected.length > 0 ? (
                    <div className="image-model-chips">
                      {llmDetected.map((ing, idx) => (
                        <span key={`llm-${ing}-${idx}`} className="image-model-chip">
                          {ing}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="detected-empty">No OpenRouter ingredients detected yet.</p>
                  )}
                </div>
              </div>
            )}
          </div>

          {results.length > 0 && (
            <div className="results-header">
              <h2 className="results-title">
                <span className="results-count">{results.length}</span>
                <span className="results-label">Recipe{results.length !== 1 ? 's' : ''} Found</span>
              </h2>
            </div>
          )}

          {results.length === 0 && !loading && !error && (
            <div className="empty-state">
              <div className="empty-illustration">
                <div className="empty-circle empty-circle-1"></div>
                <div className="empty-circle empty-circle-2"></div>
                <div className="empty-circle empty-circle-3"></div>
              </div>
              <h3 className="empty-title">Ready to cook?</h3>
              <p className="empty-text">Type in your ingredients above and discover amazing recipes</p>
            </div>
          )}

          <div className="recipes-container">
            {results.map((recipe, index) => (
              <article
                key={recipe.title}
                className="recipe-tile"
                style={{ '--delay': `${index * 50}ms` }}
              >
                <div className="tile-header">
                  <h3 className="tile-title">{recipe.title}</h3>
                  {recipe.time && (
                    <div className="tile-badge">
                      <svg className="badge-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                      </svg>
                      {recipe.time}m
                    </div>
                  )}
                </div>
                
                <div className="tile-body">
                  <div className="tile-section">
                    <span className="section-label">Ingredients</span>
                    <p className="section-text">{recipe.ingredients}</p>
                  </div>
                  
                  <div className="tile-section">
                    <span className="section-label">Instructions</span>
                    <p className="section-text">
                      {recipe.instructions.slice(0, 120)}
                      {recipe.instructions.length > 120 ? '...' : ''}
                    </p>
                  </div>
                </div>

                <div className="tile-footer">
                  {recipe.score && (
                    <div className="match-indicator">
                      <div className="match-bar">
                        <div 
                          className="match-fill" 
                          style={{ width: `${recipe.score * 100}%` }}
                        ></div>
                      </div>
                      <span className="match-text">{Math.round(recipe.score * 100)}% match</span>
                    </div>
                  )}
                  <button
                    className="tile-action"
                    onClick={() => openRecipe(recipe)}
                  >
                    View Recipe
                    <svg className="action-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <line x1="5" y1="12" x2="19" y2="12"></line>
                      <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                  </button>
                </div>
              </article>
            ))}
          </div>
        </div>
      </div>

      {/* Recipe Detail Modal */}
      {selectedRecipe && (
        <>
          <div className="modal-backdrop" onClick={closeModal}></div>
          <div className="recipe-modal">
            <button className="modal-close-btn" onClick={closeModal} aria-label="Close">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>

            <div className="recipe-modal-content">
              <div className="modal-header-section">
              <h2 className="modal-recipe-title">{selectedRecipe.title}</h2>
              {selectedRecipe.time && (
                <div className="modal-time-badge">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                  </svg>
                  {selectedRecipe.time} minutes
                </div>
              )}
            </div>

            <div className="modal-body">
              <section className="modal-section-block">
                <div className="section-header">
                  <div className="section-icon-wrapper">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M9 11l3 3L22 4"></path>
                      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
                    </svg>
                  </div>
                  <h3 className="section-heading">Ingredients</h3>
                </div>
                <div className="ingredients-grid">
                  {selectedRecipe.ingredients.split(',').map((ing, i) => (
                    <div key={i} className="ingredient-chip">
                      {ing.trim()}
                    </div>
                  ))}
                </div>
              </section>

              <section className="modal-section-block">
                <div className="section-header">
                  <div className="section-icon-wrapper">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                      <polyline points="14 2 14 8 20 8"></polyline>
                      <line x1="16" y1="13" x2="8" y2="13"></line>
                      <line x1="16" y1="17" x2="8" y2="17"></line>
                      <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                  </div>
                  <h3 className="section-heading">Instructions</h3>
                </div>
                <div className="instructions-container">
                  {selectedRecipe.instructions.split('\n').filter(step => step.trim()).map((step, i) => (
                    <div key={i} className="instruction-item">
                      <div className="instruction-number">{i + 1}</div>
                      <div className="instruction-content">{step.trim()}</div>
                    </div>
                  ))}
                </div>
              </section>

              <section className="modal-section-block">
                <div className="section-header">
                  <div className="section-icon-wrapper">
                    <svg viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                      <path d="M12 4C9.243 4 7 6.243 7 9h2c0-1.654 1.346-3 3-3s3 1.346 3 3c0 1.069-.454 1.465-1.481 2.255-.382.294-.813.626-1.226 1.038C10.981 13.604 10.995 14.897 11 15v2h2v-2.009c0-.024.023-.601.707-1.284.32-.32.682-.598 1.031-.867C15.798 12.024 17 11.1 17 9c0-2.757-2.243-5-5-5zm-1 14h2v2h-2z"/>
                    </svg>
                  </div>
                  <h3 className="section-heading">Missing Something?</h3>
                </div>
                <div className="adaptation-panel">
                  <div className="adaptation-controls">
                    <input
                      type="text"
                      className="adaptation-field"
                      placeholder="What ingredient are you missing?"
                      value={missing}
                      onChange={(e) => setMissing(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && getAdaptationAdvice()}
                      disabled={adaptLoading}
                    />
                    <button
                      className="adaptation-submit"
                      onClick={getAdaptationAdvice}
                      disabled={!missing.trim() || adaptLoading}
                    >
                      {adaptLoading ? (
                        <span className="btn-loader-small"></span>
                      ) : (
                        <>
                          <span>Get Suggestion</span>
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path d="M5 12h14M12 5l7 7-7 7"></path>
                          </svg>
                        </>
                      )}
                    </button>
                  </div>

                  {adaptedStep && (
                    <div className="adaptation-result">
                      <div className="result-header">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                          <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <span>AI Suggestion</span>
                      </div>
                      <div className="result-text">
                        {formatAdaptationText(adaptedStep).length > 0 ? (
                          <ul className="adaptation-list">
                            {formatAdaptationText(adaptedStep).map((item, index) => (
                              <li key={index}>{item}</li>
                            ))}
                          </ul>
                        ) : (
                          <p>{adaptedStep}</p>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </section>

              <section className="modal-section-block">
                <div className="section-header">
                  <div className="section-icon-wrapper">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <polygon points="23 7 16 12 23 17 23 7"></polygon>
                      <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                    </svg>
                  </div>
                  <h3 className="section-heading">Video Tutorials</h3>
                </div>
                {videosLoading ? (
                  <div className="videos-loading-state">
                    <span className="btn-loader-small"></span>
                    <span>Loading videos...</span>
                  </div>
                ) : videos.length > 0 ? (
                  <div className="videos-grid">
                    {videos.map((v, i) => (
                      <a
                        key={i}
                        href={v.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="video-card"
                      >
                        <div className="video-thumbnail">
                          <svg viewBox="0 0 24 24" fill="currentColor">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                          </svg>
                        </div>
                        <div className="video-info">
                          <span className="video-name">{v.title}</span>
                          <span className="video-link-text">Watch on YouTube</span>
                        </div>
                      </a>
                    ))}
                  </div>
                ) : (
                  <p className="no-videos-message">No video tutorials available</p>
                )}
              </section>
            </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
