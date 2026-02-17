#!/bin/bash
# Quick deployment script for mvmp package

set -e  # Exit on error

echo "🚀 MVMP Package Deployment Script"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# Function to prompt for action
prompt_continue() {
    read -p "$1 (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Aborted"
        exit 1
    fi
}

# Step 1: Clean old builds
echo "📦 Step 1: Cleaning old builds..."
rm -rf dist/ build/ *.egg-info mvmp.egg-info
echo "✓ Clean complete"
echo ""

# Step 2: Build package
echo "🔨 Step 2: Building package with uv..."
uv build
echo "✓ Build complete"
echo ""

# Show what was built
echo "📋 Built packages:"
ls -lh dist/
echo ""

# Step 3: Test locally (optional)
prompt_continue "🧪 Test the package locally before publishing?"

echo "Creating test environment..."
uv venv .test-env
source .test-env/bin/activate

echo "Installing built package..."
uv pip install dist/mvmp-*.whl

echo "Testing import..."
python -c "from mvmp import Facemarker, FacemarkerResult; print('✓ Import successful')"

echo "Testing CLI..."
mvmp --help > /dev/null && echo "✓ CLI works"

deactivate
rm -rf .test-env

echo ""
echo "✓ Local tests passed!"
echo ""

# Step 4: Choose deployment target
echo "🎯 Step 4: Choose deployment target"
echo "1) TestPyPI (recommended first)"
echo "2) Production PyPI"
echo "3) Skip deployment"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "📤 Deploying to TestPyPI..."
        echo "Note: You'll need your TestPyPI API token"
        echo ""

        uv run --with twine twine upload --repository testpypi dist/*

        echo ""
        echo "✅ Published to TestPyPI!"
        echo ""
        echo "Test installation with:"
        echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mvmp"
        echo ""
        echo "View at: https://test.pypi.org/project/mvmp/"
        ;;

    2)
        echo ""
        prompt_continue "⚠️  Deploy to PRODUCTION PyPI? This cannot be undone!"

        echo "📤 Deploying to Production PyPI..."

        uv run --with twine twine upload dist/*

        echo ""
        echo "🎉 Published to PyPI!"
        echo ""
        echo "Install with: pip install mvmp"
        echo "View at: https://pypi.org/project/mvmp/"
        ;;

    3)
        echo "Skipping deployment."
        echo "You can manually deploy later with:"
        echo "  uv run --with twine twine upload --repository testpypi dist/*  # TestPyPI"
        echo "  uv run --with twine twine upload dist/*                         # Production PyPI"
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🎊 Done!"
