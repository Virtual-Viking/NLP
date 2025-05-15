# Deploying Wordsmith SMS Analysis Suite on Vercel

This guide will walk you through deploying your Streamlit app using Streamlit Community Cloud, with a Vercel redirect pointing to your Streamlit deployment.

## Why this approach?

While Vercel is excellent for many web applications, it doesn't natively support Streamlit apps. Streamlit Community Cloud offers free hosting specifically designed for Streamlit applications. Our deployment will:

1. Host the actual app on Streamlit Community Cloud
2. Create a simple redirect app on Vercel that points to your Streamlit deployment

## Step 1: Prepare Your Repository

1. Make sure your repository has these essential files:
   - `app.py` (your main Streamlit application)
   - `requirements.txt` (dependencies)
   - `README.md` (project documentation)
   - `vercel.json` (Vercel configuration)
   - `api/index.py` (Vercel API route for redirection)

2. Ensure all your imports and dependencies are correctly listed in `requirements.txt`

## Step 2: Deploy to Streamlit Community Cloud

1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch, and main file (`app.py`)
5. Click "Deploy"
6. Wait for the deployment to complete (this may take a few minutes)
7. Note the URL of your deployed app (e.g., `https://wordsmith-sms-analysis.streamlit.app`)

## Step 3: Update the Vercel Redirect

1. Open `api/index.py`
2. Update the redirect URL to match your Streamlit app's URL:
   ```python
   # Replace this URL with your actual Streamlit app URL
   window.location.href = "https://your-app-name.streamlit.app/"
   ```
3. Save the changes and commit to your repository

## Step 4: Deploy to Vercel

1. Go to [Vercel](https://vercel.com/)
2. Sign up or log in (you can use your GitHub account)
3. Click "New Project"
4. Import your GitHub repository
5. Configure the project:
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: Leave empty (uses vercel.json)
   - Output Directory: Leave empty
6. Click "Deploy"

## Step 5: Configure Custom Domain (Optional)

1. Once deployed on Vercel, go to your project settings
2. Click on "Domains"
3. Add your custom domain and follow the verification process

## Troubleshooting

- **Deployment Failed**: Check your `requirements.txt` to ensure all dependencies are correctly specified with compatible versions
- **App Not Loading**: Verify the redirect URL in `api/index.py` matches your Streamlit app URL
- **Module Not Found Errors**: Make sure all imports in your app are included in `requirements.txt`

## Maintenance

To update your app:
1. Make changes to your code
2. Commit and push to GitHub
3. Both Streamlit Community Cloud and Vercel will automatically redeploy your app

## Resources

- [Streamlit Deployment Documentation](https://docs.streamlit.io/streamlit-cloud/get-started)
- [Vercel Python Documentation](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
